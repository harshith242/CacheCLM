// 00: why an edit costs money on a hosted API. The prompt is a train of wagons passing the provider's toll. Wagons that
// match the previous request ride at the hit price; from the first changed wagon on, every wagon pays the miss price.

function wagon(x, y, color, fill) {
  rect(x, y, 120, 70, { stroke: color, fill: fill || "#fff", fillStyle: fill ? "hachure" : "solid", hachureGap: 9 });
  ell(x + 28, y + 80, 22, 22, { stroke: INK }); ell(x + 92, y + 80, 22, 22, { stroke: INK });
  line(x + 120, y + 45, x + 140, y + 45, { strokeWidth: 2 });
}
function coin(x, y, big, color) { ell(x, y, big ? 46 : 20, big ? 46 : 20, { stroke: color, fill: "#fff", fillStyle: "solid" }); }

// The toll booth on the right, shared by both trains.
rect(1600, 160, 150, 760, { strokeWidth: 3 });
line(1590, 160, 1675, 100, { strokeWidth: 3 }); line(1675, 100, 1760, 160, { strokeWidth: 3 });
text(1675, 300, "toll", { size: 40, anchor: "middle", weight: 700 });
label(1560, 980, "provider cache", "prices.yaml", "miss = 50 × hit (DeepSeek)", { color: RED });

// Top train: a pure append. Old wagons match the cache; only the new wagon pays the miss price.
for (let i = 0; i < 9; i++) wagon(80 + i * 160, 250, BLUE, BLUE);
wagon(80 + 9 * 160 - 40, 250, ORANGE, ORANGE);
for (let i = 0; i < 9; i++) coin(150 + i * 160, 215, false, BLUE);
coin(1490, 205, true, RED);
label(80, 120, "append", "pure append", "only the new part misses", { color: ORANGE });

// Bottom train: Xiaohei swaps a wagon in the middle; from there on every wagon pays the miss price.
for (let i = 0; i < 3; i++) wagon(80 + i * 160, 640, BLUE, BLUE);
for (let i = 3; i < 9; i++) wagon(80 + i * 160, 640, RED, i === 3 ? ORANGE : RED);
for (let i = 0; i < 3; i++) coin(150 + i * 160, 605, false, BLUE);
for (let i = 3; i < 9; i++) coin(150 + i * 160, 600, true, RED);
xiaohei(500, 470, { s: 0.85, armL: [450, 520], armR: [585, 645], legL: [482, 555], legR: [520, 555] });
line(585, 645, 610, 668, { strokeWidth: 4 });      // the wrench on the swapped wagon
label(80, 830, "edit in the middle", "first change", "every later token misses", { color: RED });
label(560, 880, "cached prefix", "provider prompt cache", "matches the last request: hit price", { color: BLUE });
pointer(575, 850, 300, 735, BLUE);
label(1120, 860, "Recorder", "run.py", "logs billed + ideal hits", { color: BLUE, size: 30 });
