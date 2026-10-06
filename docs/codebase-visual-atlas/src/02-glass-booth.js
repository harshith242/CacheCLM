// 02: the CLM edit loop. Xiaohei (the model) only posts a slip into a sealed glass booth; the edited scroll is checked
// and the booth's receipt is strung back to the note on Xiaohei's wall. An over-long slip is cut off.

// The wall note (control): budget use, a nudge shown once, the over-limit warning, the last result.
rect(60, 60, 430, 220, { fill: "#fff", fillStyle: "solid" });
ell(275, 64, 14, 14, { fill: INK, fillStyle: "solid" });
text(80, 108, "Context: 7,812 of 10,000 (78%)", { size: 26 });
text(80, 148, "Your context is over 75% full.", { size: 26, color: ORANGE });
ell(440, 140, 64, 36, { stroke: ORANGE, strokeWidth: 1.6 }); text(440, 148, "once", { size: 20, color: ORANGE, anchor: "middle" });
text(80, 190, "OVER LIMIT: free 1,200 tokens", { size: 26, color: RED });
text(80, 232, "Result of your last command: ...", { size: 26, color: BLUE });
label(60, 330, "control", "arms.py", "nudge once · OVER LIMIT every call");

// The desk, Xiaohei writing the slip.
line(150, 720, 560, 720, { strokeWidth: 3 }); line(180, 720, 180, 840); line(530, 720, 530, 840);
xiaohei(330, 600, { armL: [270, 640], armR: [420, 690], legL: [305, 718], legR: [355, 718] });
line(420, 690, 445, 650, { strokeWidth: 3 });      // the pen
poly([[430, 690], [770, 530], [800, 600], [465, 745]], { fill: "#fff", fillStyle: "solid" });  // the slip
text(490, 694, "```text  notes...", { size: 22, color: BLUE, rotate: -25 });
text(510, 724, "```bash  python3 -c ...", { size: 22, rotate: -25 });
label(560, 800, "parse_command", "arms.py", "text block → new.txt, bash → command");

// An over-long slip curling to the floor, cut by red scissors.
curve([[200, 722], [215, 820], [185, 900], [230, 990]], { strokeWidth: 2 });
curve([[240, 722], [258, 820], [228, 900], [272, 990]], { strokeWidth: 2 });
for (let i = 0; i < 7; i++) line(212 + (i % 2) * 4, 760 + i * 30, 238 + (i % 2) * 4, 760 + i * 30, { stroke: "#999", strokeWidth: 1.2 });
scissors(222, 905, 0.8, RED);
label(300, 950, "cut_off", "edit_phase", "too long: the phase ends", { color: RED });

// The glass booth: allow-list clipboard at the door, padlock, scroll edited by a mechanical arm.
rect(780, 300, 420, 520, { strokeWidth: 3 });
for (let i = 0; i < 4; i++) line(820 + i * 90, 330, 860 + i * 90, 400, { stroke: "#bbb", strokeWidth: 1.4 });
ell(990, 282, 44, 40, { strokeWidth: 2.4 }); rect(966, 282, 48, 40, { fill: INK, fillStyle: "solid" });  // padlock
line(770, 555, 770, 600, { strokeWidth: 5 });       // slot
curve([[790, 568], [760, 572], [745, 574]], { stroke: ORANGE, strokeWidth: 3 });
rect(640, 360, 110, 140, { fill: "#fff", fillStyle: "solid" });
for (const [i, w] of ["sed", "awk", "grep", "python3"].entries()) text(655, 395 + i * 28, w, { size: 22 });
label(560, 150, "sandbox.check", "sandbox.py", "allow-list at the door");
pointer(640, 250, 680, 355);
// scroll ctx.txt
rect(850, 520, 280, 150, { fill: "#fff", fillStyle: "solid" });
ell(850, 595, 26, 150); ell(1130, 595, 26, 150);
text(990, 560, "ctx.txt", { size: 30, anchor: "middle" });
for (let i = 0; i < 3; i++) line(880, 590 + i * 22, 1100 - i * 30, 590 + i * 22, { stroke: "#999", strokeWidth: 1.3 });
path("M 1150 320 L 1110 420 L 1040 470 L 1060 630", { strokeWidth: 4 });   // mechanical arm
ell(1110, 420, 18, 18, { fill: INK, fillStyle: "solid" }); ell(1040, 470, 18, 18, { fill: INK, fillStyle: "solid" });
rect(860, 700, 110, 80, { stroke: BLUE, fill: "#fff", fillStyle: "solid" });
text(915, 748, "new.txt", { size: 24, color: BLUE, anchor: "middle" });
label(1000, 130, "sandbox.run", "sandbox.py", "sandbox-exec jail · 10 s");
pointer(1050, 220, 1000, 268);

// Out of the booth: the emptied check, then the gate.
curve([[1200, 590], [1250, 580], [1290, 600]], { stroke: ORANGE, strokeWidth: 3 });
arrowHead(1290, 600, 0.3);
rect(1300, 560, 120, 80, { fill: "#fff", fillStyle: "solid" });
ell(1400, 540, 60, 60, { stroke: RED }); line(1420, 562, 1445, 590, { stroke: RED, strokeWidth: 4 });
label(1260, 720, "emptied?", "edit_phase", "headers only → roll back", { color: RED });
curve([[1440, 600], [1520, 590], [1580, 600]], { stroke: ORANGE, strokeWidth: 3 });
arrowHead(1580, 600, 0.2);
line(1680, 480, 1680, 640, { strokeWidth: 3 }); line(1600, 500, 1760, 500, { strokeWidth: 3 });
path("M 1590 500 L 1570 560 L 1630 560 Z"); path("M 1750 500 L 1730 560 L 1790 560 Z");
line(1640, 645, 1720, 645, { strokeWidth: 3 });
label(1600, 720, "decide", "gate.py", "pays off? (next image)");

// The receipt strung back to the wall note: the result of the last command.
curve([[1680, 470], [1600, 250], [1250, 40], [700, 35], [495, 230]], { stroke: BLUE, strokeWidth: 2.2, strokeLineDash: [10, 7] });
arrowHead(495, 230, 2.4, BLUE);
label(1300, 300, "last", "edit_phase", "result fed into the next note", { color: BLUE });
