/* 2D yard viewer. WAVE UX-A may later drop a three.js mesh viewer at
   /static/ux_a/index.html — deep-link that instead of adding a second WebGL stack. */
(function (global) {
  const UX_A = "/static/ux_a/index.html";

  function poly(points) {
    return points.map((p) => `${p[0]},${p[1]}`).join(" ");
  }

  function drawYard(svg, payload) {
    const yard = payload.yard || {};
    const status = payload.status || {};
    const coverage = payload.coverage || {};
    const w = Number(yard.width_m || 16);
    const h = Number(yard.height_m || 12);
    const keepIn = yard.keep_in || [];
    const keepOut = yard.keep_out || [];
    const pose = status.pose || { x: 1, y: 1, theta: 0 };
    const rows = Number(coverage.rows || 0);
    const cols = Number(coverage.cols || 0);
    const values = coverage.values || [];
    let cells = "";
    if (rows && cols && values.length === rows * cols) {
      const cw = w / cols;
      const ch = h / rows;
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const v = values[r * cols + c];
          let fill = "none";
          if (v >= 0.5) fill = "rgba(210, 190, 80, 0.55)";
          else if (v < 0) fill = "rgba(40, 30, 20, 0.35)";
          if (fill !== "none") {
            cells += `<rect x="${c * cw}" y="${r * ch}" width="${cw}" height="${ch}" fill="${fill}"/>`;
          }
        }
      }
    }
    const hx = Number((yard.home || {}).x || 1);
    const hy = Number((yard.home || {}).y || 1);
    const th = Number(pose.theta || 0);
    const px = Number(pose.x || 0);
    const py = Number(pose.y || 0);
    const nose = [px + 0.38 * Math.cos(th), py + 0.38 * Math.sin(th)];
    const left = [px - 0.2 * Math.sin(th), py + 0.2 * Math.cos(th)];
    const right = [px + 0.2 * Math.sin(th), py - 0.2 * Math.cos(th)];
    const outs = keepOut
      .map(
        (ring) =>
          `<polygon points="${poly(ring)}" fill="rgba(200,40,40,0.35)" stroke="#c82828" stroke-width="0.08"/>`
      )
      .join("");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.innerHTML = `
      <g transform="translate(0 ${h}) scale(1 -1)">
        <rect x="0" y="0" width="${w}" height="${h}" fill="#2e8c3a"/>
        ${cells}
        <polygon points="${poly(keepIn)}" fill="none" stroke="#ffcc33" stroke-width="0.12"/>
        ${outs}
        <circle cx="${hx}" cy="${hy}" r="0.22" fill="#8be0a0" stroke="#06210c" stroke-width="0.05"/>
        <polygon points="${nose[0]},${nose[1]} ${left[0]},${left[1]} ${right[0]},${right[1]}" fill="#111" stroke="#fff" stroke-width="0.04"/>
      </g>`;
  }

  async function uxAAvailable() {
    try {
      const res = await fetch(UX_A, { method: "HEAD" });
      return res.ok;
    } catch (_err) {
      return false;
    }
  }

  global.JimsViewer = {
    drawYard,
    uxAHref: UX_A,
    uxAAvailable,
  };
})(window);
