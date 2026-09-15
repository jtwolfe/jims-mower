"""ROS 2-lite runtime bridge: stdlib multiprocessing queues.

CI and laptops do not have ROS 2. This bridge shuttles versioned contract
payloads (and optional gym observations) between processes with
``multiprocessing.Queue`` — no ZMQ, no gRPC. Record / replay uses the same
``EpisodeReader`` frames the gym logger already writes.

Optional ROS 2 node stubs live in ``jims_mower.runtime.ros2_stubs`` and
are documented under the ``[ros2]`` extra. They are not imported here.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.contract import CONTRACT_VERSION, MESSAGE_KINDS, validate_payload
from jims_mower.episode import EpisodeReader
from jims_mower.runtime.bus import BusMessage, InProcessBus
from jims_mower.runtime.drivers import SensorRig, cameras_from_obs, gps_from_obs, imu_from_obs, tof_from_obs

_SENTINEL = None


def _worker(inbox: mp.Queue, outbox: mp.Queue) -> None:
    """Child process: gym-shaped obs dicts → contract payloads."""
    rig = SensorRig()
    while True:
        item = inbox.get()
        if item is _SENTINEL:
            outbox.put(_SENTINEL)
            return
        if not isinstance(item, dict):
            continue
        stamp = float(item.get("stamp_s") or 0.0)
        obs = item.get("obs") or item
        msgs = rig.publish(obs, stamp_s=stamp)
        outbox.put(
            {
                "n": len(msgs),
                "messages": [
                    {"kind": m.kind, "payload": m.payload, "bus": m.bus} for m in msgs
                ],
            }
        )


class MultiprocessBridge:
    """Parent publishes observations; a child runs the fake sensor rig.

    ``use_process=False`` keeps everything on the caller thread (unit tests).
    ``use_process=True`` starts one ``spawn`` worker so the gym process can
    stay separate from the driver process — the same split an Orin runtime
    would use without pulling in ROS 2.
    """

    def __init__(self, *, use_process: bool = False, ctx: Optional[mp.context.BaseContext] = None) -> None:
        self.use_process = bool(use_process)
        self._ctx = ctx or mp.get_context("spawn")
        self.local = SensorRig()
        self._inbox: Optional[mp.Queue] = None
        self._outbox: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self.n_published = 0

    @property
    def bus(self) -> InProcessBus:
        return self.local.bus

    def start(self) -> None:
        if not self.use_process or self._proc is not None:
            return
        self._inbox = self._ctx.Queue()
        self._outbox = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker,
            args=(self._inbox, self._outbox),
            name="jims-mower-bridge",
            daemon=True,
        )
        self._proc.start()

    def close(self) -> None:
        if self._inbox is not None:
            try:
                self._inbox.put(_SENTINEL)
            except (OSError, ValueError):
                pass
        if self._outbox is not None:
            try:
                while True:
                    item = self._outbox.get(timeout=0.2)
                    if item is _SENTINEL:
                        break
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None
        self._inbox = None
        self._outbox = None

    def __enter__(self) -> "MultiprocessBridge":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def publish_obs(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[BusMessage]:
        """Publish one gym / recorded observation through the fake drivers."""
        self.n_published += 1
        if not self.use_process or self._inbox is None or self._outbox is None:
            return self.local.publish(obs, stamp_s=stamp_s)
        self._inbox.put({"obs": obs, "stamp_s": float(stamp_s)})
        reply = self._outbox.get(timeout=10.0)
        if reply is _SENTINEL or not isinstance(reply, dict):
            return []
        out: list[BusMessage] = []
        for item in reply.get("messages") or []:
            kind = str(item["kind"])
            payload = validate_payload(kind, item["payload"])
            out.append(
                self.local.bus.publish(kind, payload, bus=str(item.get("bus") or "mp"))
            )
        return out

    def publish_messages(self, messages: list[dict[str, Any]]) -> list[BusMessage]:
        """Enqueue already-versioned contract dicts (record/replay sidecars)."""
        out: list[BusMessage] = []
        for item in messages:
            kind = str(item["kind"])
            payload = item.get("payload") or item
            out.append(self.local.bus.publish(kind, payload, bus=str(item.get("bus") or "replay")))
        return out


def messages_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[dict[str, Any]]:
    """Pure helper: gym obs → list of ``{kind, payload, bus}`` (no queues)."""
    imu = imu_from_obs(obs, stamp_s=stamp_s)
    gps = gps_from_obs(obs, stamp_s=stamp_s)
    tof = tof_from_obs(obs, stamp_s=stamp_s)
    cams = cameras_from_obs(obs, stamp_s=stamp_s)
    items = [
        {"kind": "ImuSample", "payload": imu.to_dict(), "bus": "i2c"},
        {"kind": "GpsFix", "payload": gps.to_dict(), "bus": "uart"},
        {"kind": "TofArray", "payload": tof.to_dict(), "bus": "i2c"},
    ]
    for frame in cams:
        items.append({"kind": "CameraFrame", "payload": frame.to_dict(), "bus": "csi"})
    return items


def replay_through_bridge(
    episode_dir: Union[str, Path],
    *,
    use_process: bool = False,
    out_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Replay a recorded episode into the fake drivers (no physics).

    Each logged observation is published as contract messages. Compatible
    with ``jims-mower-record`` directories (manifest + frames/*.npz).
    """
    reader = EpisodeReader(episode_dir)
    kinds: dict[str, int] = {k: 0 for k in MESSAGE_KINDS}
    n_obs = 0
    with MultiprocessBridge(use_process=use_process) as bridge:
        if reader.reset_obs is not None:
            for msg in bridge.publish_obs(reader.reset_obs, stamp_s=0.0):
                kinds[msg.kind] = kinds.get(msg.kind, 0) + 1
            n_obs += 1
        for i, rec in enumerate(reader.steps, start=1):
            obs = rec.get("obs") or {}
            for msg in bridge.publish_obs(obs, stamp_s=float(i) * float(reader.manifest.get("dt") or 0.1)):
                kinds[msg.kind] = kinds.get(msg.kind, 0) + 1
            n_obs += 1
    summary = {
        "mode": "bridge",
        "source": str(Path(episode_dir)),
        "contract_version": CONTRACT_VERSION,
        "n_observations": n_obs,
        "n_messages": sum(kinds.values()),
        "kinds": kinds,
        "use_process": bool(use_process),
        "not_a_benchmark": True,
        "fps_claim": None,
    }
    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay a recorded episode through fake I2C/UART/CSI drivers"
    )
    p.add_argument("episode", type=Path, help="directory from jims-mower-record")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--process",
        action="store_true",
        help="run the driver rig in a spawn child (default: in-process)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = replay_through_bridge(args.episode, use_process=args.process, out_path=args.out)
    print(
        f"Bridge replay {summary['n_observations']} obs → "
        f"{summary['n_messages']} contract messages "
        f"(not a board FPS claim)"
    )


if __name__ == "__main__":
    main()
