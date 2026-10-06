// 06: the same harness on two books (EventQA, 41K tokens each, about 4x the budget). Each arm is a jar filled to its
// accuracy, with its billed cost on a tag; skill's jars have cracked lids because it stopped reading on overflow.
const H = 300;   // jar height for accuracy 1.0
const shelves = [
  ["book 1", "row 12 · 33 questions", 170, 470,
   [["summary", 0.91, 0.071, false], ["clm", 0.91, 0.176, false], ["gate", 0.91, 0.098, false], ["skill", 0.94, 0.245, true]]],
  ["book 2", "row 13 · 30 questions", 170, 960,
   [["summary", null, null, false], ["clm", 0.97, 0.147, false], ["gate", 0.90, 0.158, false], ["skill", 0.87, 0.081, true]]],
];
const COLORS = { summary: INK, clm: ORANGE, gate: BLUE, skill: RED };

for (const [name, where, x0, base, jars] of shelves) {
  line(x0 - 40, base, x0 + 1140, base, { strokeWidth: 4 });                 // the shelf plank
  line(x0 - 20, base, x0 - 20, base + 30); line(x0 + 1120, base, x0 + 1120, base + 30);
  label(x0 - 40, base - H - 90, name, where, null, { size: 40 });
  jars.forEach(([arm, acc, usd, stopped], k) => {
    const x = x0 + 120 + k * 270, w = 150;
    if (acc === null) {                                                       // summary was not run on book 2
      rect(x - w / 2, base - H, w, H, { stroke: "#bbb", strokeLineDash: [8, 8] });
      text(x, base - H / 2, "not run", { size: 26, anchor: "middle", color: "#999" });
    } else {
      rect(x - w / 2, base - acc * H, w, acc * H, { stroke: COLORS[arm], fill: COLORS[arm], fillStyle: "hachure", hachureGap: 9 });
      rect(x - w / 2, base - H, w, H, { strokeWidth: 2.4 });
      // price tag on a string
      line(x + w / 2, base - H + 20, x + w / 2 + 30, base - H + 60, { strokeWidth: 1.2 });
      poly([[x + w / 2 + 20, base - H + 60], [x + w / 2 + 110, base - H + 60], [x + w / 2 + 110, base - H + 100],
            [x + w / 2 + 20, base - H + 100]], { fill: "#fff", fillStyle: "solid" });
      text(x + w / 2 + 65, base - H + 89, `$${usd.toFixed(3)}`, { size: 22, anchor: "middle", font: "Patrick Hand" });
    }
    // lid; a cracked lid marks an arm that stopped reading early
    line(x - w / 2 - 8, base - H - 8, x + w / 2 + 8, base - H - 8, { strokeWidth: 4, stroke: stopped ? RED : INK });
    if (stopped) path(`M ${x - 10} ${base - H - 20} L ${x + 4} ${base - H - 4} L ${x - 6} ${base - H + 10} L ${x + 8} ${base - H + 26}`,
                      { stroke: RED, strokeWidth: 2.4 });
    text(x, base + 45, acc === null ? arm : `${arm}  ${acc.toFixed(2)}`, { size: 32, anchor: "middle", weight: 700,
                                                                         color: COLORS[arm] });
  });
}

// Xiaohei, standing between the shelves, holds the gate's two price tags up to compare them.
xiaohei(1110, 562, { s: 0.9, armL: [990, 290], armR: [990, 700], legL: [1092, 640], legR: [1128, 640] });
ell(985, 250, 120, 120, { stroke: BLUE, strokeWidth: 2.4 }); ell(985, 740, 120, 120, { stroke: BLUE, strokeWidth: 2.4 });
label(1570, 330, "gate", "decide", "cheaper on book 1, not on book 2", { color: BLUE });
label(1570, 760, "stopped early", "overflow_stop", "skill: 7, then 13 parts unread", { color: RED });
