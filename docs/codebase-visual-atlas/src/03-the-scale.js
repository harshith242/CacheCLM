// 03: the cache-aware gate. The context is a long strip whose front is billed at the cache-hit price. Xiaohei marks the
// first change; everything after it turns red (billed once at the miss price). A scale weighs that against the saving.

// The strip: cached front (blue), the first-change mark (orange), the rebilled tail (red).
rect(120, 200, 940, 90, { fill: BLUE, fillStyle: "hachure", hachureGap: 14, stroke: INK, strokeWidth: 2.4 });
rect(1060, 200, 740, 90, { fill: RED, fillStyle: "hachure", hachureGap: 9, stroke: INK, strokeWidth: 2.4 });
ell(470, 245, 190, 62, { stroke: BLUE, fill: "#fff", fillStyle: "solid" });
text(470, 255, "cached", { size: 34, color: BLUE, anchor: "middle", weight: 700 });
line(1060, 180, 1060, 312, { stroke: ORANGE, strokeWidth: 6 });
xiaohei(1110, 80, { s: 0.85, armL: [1066, 175], armR: [1160, 70], legL: [1092, 196], legR: [1130, 196] });
line(1066, 175, 1060, 196, { strokeWidth: 4 });      // the pen
label(640, 60, "first_change", "gate.py", "where the edit starts", { color: ORANGE });
pointer(820, 110, 1050, 190, ORANGE);
label(1380, 80, "rebilled", "gate.py", "every token after it, once", { color: RED });
pointer(1440, 160, 1500, 205, RED);

// Deleted text falling off the strip into a bin.
for (const [x, y, t] of [[1200, 330, 12], [1240, 370, -18], [1180, 395, 30]]) page(x, y, 40, 26, { tilt: t, lines: 0, color: RED });
path("M 1160 430 L 1180 540 L 1290 540 L 1310 430", { strokeWidth: 2.4 });
line(1150, 430, 1320, 430, { strokeWidth: 2.4 });
label(1340, 450, "deleted", "gate.py", "tokens the edit frees");

// The scale: saving on the left pan, cost on the right; the beam tips to the heavier side.
line(960, 640, 960, 960, { strokeWidth: 4 });
line(880, 965, 1040, 965, { strokeWidth: 4 });
line(640, 590, 1280, 640, { strokeWidth: 4 });        // beam, tipped right: cost wins this time
poly([[960, 616], [936, 645], [984, 645]], { fill: INK, fillStyle: "solid" });
line(640, 590, 580, 720); line(640, 590, 700, 720); ell(640, 725, 150, 26);
line(1280, 640, 1220, 770); line(1280, 640, 1340, 770); ell(1280, 775, 150, 26);
for (let i = 0; i < 2; i++) ell(640, 708 - i * 10, 46, 14, { stroke: BLUE, fill: "#fff", fillStyle: "solid" });
for (let i = 0; i < 5; i++) ell(1280 + (i % 2) * 10, 758 - i * 10, 46, 14, { stroke: RED, fill: "#fff", fillStyle: "solid" });
label(470, 800, "saving", "deleted × turns_left × hit", "every turn still to come", { color: BLUE });
label(1180, 850, "cost", "rebilled × (miss − hit)", "paid once, on the next call", { color: RED });

// Price tag, the free append, and the overflow lever.
poly([[80, 400], [330, 400], [370, 450], [330, 500], [80, 500]], { fill: "#fff", fillStyle: "solid" });
ell(345, 450, 12, 12);
text(100, 440, "hit   $0.006 / M", { size: 28, color: BLUE });
text(100, 482, "miss  $0.30 / M", { size: 28, color: RED });
label(80, 560, "prices.yaml", "configs", "DeepSeek list prices");
rect(80, 730, 220, 40, { fill: BLUE, fillStyle: "hachure", hachureGap: 12 });
rect(300, 730, 56, 40, { stroke: ORANGE, fill: ORANGE, fillStyle: "hachure", hachureGap: 8 });
text(372, 760, "+", { size: 40, color: ORANGE });
label(80, 830, "pure append", "appended_only", "free: no cached token touched", { color: ORANGE });
line(1640, 780, 1720, 560, { strokeWidth: 5 });
ell(1722, 552, 34, 34, { fill: ORANGE, fillStyle: "solid", stroke: ORANGE });
path("M 1590 790 Q 1640 760 1700 790", { strokeWidth: 3 });
label(1520, 860, "overflow", "decide(overflow=True)", "over the limit: any shrink passes", { color: ORANGE });
