"""Latency harness reports host timings; no FPS claims."""

from __future__ import annotations

from jims_mower.latency import LatencyDelays, LatencyHarness, ORIN_NANO_CLASS_NOTES


def test_scorecard_not_a_benchmark() -> None:
    delays = LatencyDelays.from_ms(4.0, 4.0, 4.0)
    harness = LatencyHarness(delays)
    for _ in range(3):
        harness.start_cycle()
        harness.after_camera()
        harness.after_plan()
        harness.after_cmd()
    card = harness.scorecard()
    assert card["not_a_benchmark"] is True
    assert card["fps_claim"] is None
    assert "Orin Nano" in card["platform_assumption"]
    assert "FPS" in card["notes"]
    assert "FPS" in ORIN_NANO_CLASS_NOTES
    timings = card["timings_ms"]["camera_to_cmd"]
    assert timings["n"] == 3
    # Two injected gaps (plan + cmd) at 4 ms each, plus host overhead.
    assert timings["mean_ms"] >= 8.0
    assert card["injected_delay_s"]["camera"] == 0.004
