// 04: the four arms as a four-scene comic, one crate each, plus the two reference crates.

function crate(x, y, w = 240, h = 220) {
  rect(x, y, w, h, { strokeWidth: 2.8 });
  line(x, y, x + 36, y - 30); line(x + w, y, x + w + 36, y - 30); line(x + 36, y - 30, x + w + 36, y - 30);
  line(x + w + 36, y - 30, x + w + 36, y + h - 30); line(x + w, y + h, x + w + 36, y + h - 30);
}

// A. summary: the harness press comes down at 75%; Xiaohei writes the one summary brick.
crate(110, 430);
line(230, 150, 230, 330, { strokeWidth: 4 }); line(170, 150, 290, 150, { strokeWidth: 4 });
rect(120, 330, 240, 26, { fill: INK, fillStyle: "hachure", hachureGap: 6 });
text(300, 315, "75%", { size: 30, color: RED });
rect(140, 560, 180, 66, { fill: "#888", fillStyle: "cross-hatch", hachureGap: 10 });
text(230, 604, "summary", { size: 26, anchor: "middle", weight: 700 });
page(150, 470, 70, 50, { lines: 2 }); page(240, 470, 70, 50, { lines: 2 });
xiaohei(420, 500, { s: 0.8, armL: [325, 585], armR: [460, 470], legL: [400, 585], legR: [435, 585] });
label(110, 720, "summary", "summary_step", "harness compacts at 75%");

// B. clm: Xiaohei edits freely with scissors.
crate(570, 430);
for (let i = 0; i < 4; i++) page(600 + i * 40, 540 - i * 18, 90, 60, { tilt: i * 4 - 6, lines: 2 });
xiaohei(830, 330, { s: 0.8, armL: [760, 420], armR: [880, 300], legL: [812, 400], legR: [848, 400] });
scissors(740, 430, 0.8);
page(690, 330, 44, 30, { tilt: 30, lines: 0 }); page(650, 370, 40, 28, { tilt: -25, lines: 0 });
label(570, 720, "clm", "edit_phase", "the model edits freely");

// C. gate: the scale stands at the crate's mouth; a rejected slip lies stamped on the floor.
crate(1030, 430);
line(1150, 300, 1150, 405, { strokeWidth: 3 }); line(1090, 310, 1210, 300, { strokeWidth: 3 });
path("M 1080 310 L 1065 350 L 1110 350 Z"); path("M 1200 300 L 1185 340 L 1230 340 Z");
xiaohei(1320, 360, { s: 0.8, armL: [1240, 330], armR: [1370, 380], legL: [1300, 440], legR: [1338, 440] });
page(1205, 300, 46, 34, { tilt: -10, lines: 0 });
page(1290, 620, 90, 50, { tilt: 8, lines: 2 });
line(1300, 615, 1370, 680, { stroke: RED, strokeWidth: 4 }); line(1370, 615, 1300, 680, { stroke: RED, strokeWidth: 4 });
for (let i = 0; i < 3; i++) page(1060 + i * 50, 540 - i * 14, 90, 60, { tilt: i * 5 - 4, lines: 2 });
label(1030, 720, "gate", "decide, enforced", "the scale can say no");

// D. skill: Xiaohei reads a recipe card and writes an event log.
crate(1490, 430);
rect(1560, 160, 170, 120, { fill: "#fff", fillStyle: "solid", stroke: BLUE });
text(1645, 200, "recipe", { size: 28, color: BLUE, anchor: "middle", weight: 700 });
for (let i = 0; i < 3; i++) line(1580, 225 + i * 18, 1710 - i * 20, 225 + i * 18, { stroke: BLUE, strokeWidth: 1.3 });
xiaohei(1780, 330, { s: 0.8, armL: [1725, 270], armR: [1700, 540], legL: [1762, 400], legR: [1798, 400] });
rect(1560, 500, 160, 120, { fill: "#fff", fillStyle: "solid" });
for (let i = 0; i < 4; i++) text(1575, 532 + i * 26, `${i + 1}. —————`, { size: 20 });
label(1490, 720, "skill", "SKILL in system()", "follows a recipe card", { color: BLUE });

// Footer: the gate's verdict for clm and skill is only logged; the two reference crates bracket everything.
line(560, 900, 1500, 900, { stroke: BLUE, strokeWidth: 1.6, strokeLineDash: [8, 8] });
label(560, 950, "UNGATED", "arms.py", "clm, skill: verdict logged, never enforced", { color: BLUE, size: 30 });
rect(110, 880, 120, 100, { strokeWidth: 2.4 });
label(250, 910, "none", "REFERENCES", "no context: floor", { size: 30 });
rect(1600, 880, 120, 100, { strokeWidth: 2.4 });
for (let i = 0; i < 5; i++) page(1590 + i * 22, 850 - i * 16, 60, 40, { tilt: i * 9 - 18, lines: 0 });
label(1740, 910, "full", "REFERENCES", "whole text: ceiling", { size: 30 });
