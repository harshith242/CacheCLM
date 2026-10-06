// 01: one sample through one arm. Book cut into parts, parts dropped into a fixed-size crate one at a time,
// Xiaohei trims between parts, then the lidded crate faces the questions and a receipt tape feeds the report.

// The book and the cutter (chunk_text).
rect(70, 430, 150, 230, { fill: "#fff", fillStyle: "solid" });
line(92, 432, 92, 658);
for (let i = 0; i < 7; i++) line(225, 445 + i * 30, 238, 450 + i * 30, { strokeWidth: 1.4 });
line(150, 380, 300, 470, { strokeWidth: 4 });            // blade handle
poly([[250, 440], [330, 488], [300, 505], [228, 462]], { fill: "#fff", fillStyle: "solid" });
label(60, 770, "chunk_text", "mab.py", "cuts the text into parts");
pointer(140, 750, 200, 668);

// Parts travelling along the primary path into the crate.
curve([[260, 520], [360, 380], [520, 300], [700, 330], [800, 420]], { stroke: ORANGE, strokeWidth: 3, strokeLineDash: [14, 10] });
arrowHead(800, 420, 0.95);
page(300, 400, 64, 84, { tilt: -14 });
page(430, 300, 64, 84, { tilt: 9 });
page(590, 270, 64, 84, { tilt: -4 });
label(330, 200, "stream", "run.py", "one part at a time", { color: ORANGE });

// The crate: fixed size, capacity line, pinned task card, stacked parts.
rect(700, 420, 420, 400, { strokeWidth: 3 });
line(700, 420, 760, 370, { strokeWidth: 2.4 }); line(1120, 420, 1180, 370, { strokeWidth: 2.4 });
line(760, 370, 1180, 370, { strokeWidth: 2.4 }); line(1180, 370, 1180, 770, { strokeWidth: 2.4 }); line(1120, 820, 1180, 770);
line(712, 470, 1108, 470, { stroke: RED, strokeWidth: 1.6, strokeLineDash: [8, 8] });
text(1100, 462, "budget", { size: 24, color: RED, anchor: "end" });
rect(728, 490, 130, 78, { fill: "#fff", fillStyle: "solid" });
text(793, 538, "task", { size: 30, anchor: "middle" });
ell(793, 492, 14, 14, { fill: INK, fillStyle: "solid" });       // the pin: never edited
for (let i = 0; i < 5; i++) page(760 + (i % 2) * 120 + i * 14, 590 + i * 34, 150, 70, { tilt: (i % 2 ? 3 : -2), lines: 3 });
label(700, 880, "context box", "ctxfile.py", "task pinned + editable body");

// Xiaohei on the rim, leaning in with scissors: the arm's policy between parts.
xiaohei(1060, 300, { armL: [985, 520], armR: [1110, 330], legL: [1040, 390], legR: [1085, 372] });
scissors(1000, 540, 0.9);
label(1190, 140, "edit_phase · summary_step", "arms.py", "the arm's policy, between parts");
pointer(1240, 230, 1060, 255);

// Forced cut of the oldest lines (fit).
line(640, 700, 760, 830, { stroke: RED, strokeWidth: 4 });
poly([[600, 680], [650, 655], [700, 720], [655, 740]], { stroke: RED, fill: "#fff", fillStyle: "solid" });
for (let i = 0; i < 3; i++) line(560 - i * 30, 860 + i * 22, 610 - i * 30, 870 + i * 22, { stroke: RED, strokeWidth: 1.6 });
label(330, 930, "fit", "arms.py", "forced cut of the oldest lines", { color: RED });

// After the last part: the crate is lidded and the questions arrive.
curve([[1190, 640], [1250, 610], [1310, 640]], { stroke: ORANGE, strokeWidth: 3 });
arrowHead(1310, 640, 0.4);
rect(1330, 540, 230, 220, { strokeWidth: 3 });
rect(1318, 520, 254, 26, { fill: INK, fillStyle: "hachure", hachureGap: 7 });  // the lid
for (const [x, y, t] of [[1620, 520, -6], [1700, 560, 5], [1780, 515, -3]]) {
  page(x, y, 70, 92, { tilt: t, lines: 0 });
  text(x + 35, y + 64, "?", { size: 54, anchor: "middle", color: ORANGE });
}
label(1600, 400, "answer_all", "run.py", "questions on the frozen box", { color: ORANGE });

// Each call is metered; the tape runs into the report ledger.
ell(1445, 680, 64, 64);
curve([[1182, 745], [1290, 780], [1413, 690]], { stroke: BLUE, strokeWidth: 1.8, strokeLineDash: [6, 6] });  // editing calls too
line(1445, 680, 1468, 660, { stroke: BLUE, strokeWidth: 2.4 });
curve([[1445, 712], [1430, 800], [1520, 860], [1650, 840], [1720, 900]], { stroke: BLUE, strokeWidth: 2.2 });
for (let i = 0; i < 6; i++) line(1450 + i * 45, 820 + (i % 3) * 14, 1462 + i * 45, 830 + (i % 3) * 14, { stroke: BLUE, strokeWidth: 1.2 });
label(1180, 850, "Recorder", "run.py", "logs every call: billed + ideal hits", { color: BLUE });
rect(1720, 880, 150, 100, { fill: "#fff", fillStyle: "solid" });
line(1795, 880, 1795, 980);
label(1600, 1030, "write_report", null, null, { size: 32 });
text(1785, 1030, "report.py", { size: 22, color: "#555", font: "Patrick Hand" });
