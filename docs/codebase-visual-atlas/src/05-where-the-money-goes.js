// 05: where an editing arm's money goes (book 1, DeepSeek prices, from the run logs). One coin tower per arm, sliced
// bottom-up into cache hits (blue), cache misses (red) and output tokens (orange); 2,400 px per dollar.
const SCALE = 2400, BASE = 900, W = 190;
const towers = [
  ["summary", 330, 0.001, 0.040, 0.030, 0.071],   // hits, misses, output, billed total (unrounded sum)
  ["clm", 700, 0.001, 0.044, 0.130, 0.176],
  ["gate", 1070, 0.002, 0.025, 0.071, 0.098],
  ["skill", 1440, 0.003, 0.055, 0.187, 0.245],
];

function segment(x, top, h, color) {
  rect(x - W / 2, top, W, h, { stroke: color, fill: color, fillStyle: "hachure", hachureGap: 8 });
  for (let y = top + 16; y < top + h - 6; y += 26) ell(x, y, W - 30, 14, { stroke: color, strokeWidth: 1.2 });  // coin edges
}

line(120, BASE, 1700, BASE, { strokeWidth: 3 });
const tops = {};
for (const [arm, x, hits, misses, output, total] of towers) {
  let y = BASE;
  for (const [usd, color] of [[hits, BLUE], [misses, RED], [output, ORANGE]]) {
    const h = Math.max(usd * SCALE, 8);   // hits are so small they get a minimum sliver
    segment(x, y - h, h, color);
    y -= h;
  }
  tops[arm] = y;
  text(x, BASE + 50, arm, { size: 36, anchor: "middle", weight: 700 });
  text(x, BASE + 88, `$${total.toFixed(3)}`, { size: 28, anchor: "middle", color: "#555", font: "Patrick Hand" });
}

// Xiaohei hauls the orange sack of edit replies onto the clm tower.
const cx = 700, ct = tops.clm;
xiaohei(cx, ct - 110, { armL: [cx - 50, ct - 215], armR: [cx + 50, ct - 215], legL: [cx - 22, ct], legR: [cx + 22, ct] });
poly([[cx - 90, ct - 330], [cx + 80, ct - 340], [cx + 95, ct - 215], [cx - 80, ct - 210]],
     { stroke: ORANGE, fill: ORANGE, fillStyle: "cross-hatch", hachureGap: 10 });   // the sack, held overhead
line(cx - 5, ct - 338, cx, ct - 370, { stroke: ORANGE, strokeWidth: 3 });
label(cx + 130, ct - 330, "edit replies", "phase: edit", "the model writing its edits", { color: ORANGE });

// Legend slices, pointing at the skill tower.
label(1600, 300, "output tokens", "$1.20 / M", "67–76% of an editing arm's bill", { color: ORANGE });
pointer(1600, 330, 1540, 420, ORANGE);
label(1600, 700, "cache misses", "$0.30 / M", "close to summary's", { color: RED });
pointer(1600, 720, 1540, 840, RED);
label(140, 300, "cache hits", "$0.006 / M", "almost free", { color: BLUE });
pointer(240, 380, 230, 893, BLUE);
