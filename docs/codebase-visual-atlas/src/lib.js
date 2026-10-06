// Hand-drawn helpers on top of rough.js: wobbly lines, labels in a handwriting font, and Xiaohei.
const INK = "#1b1b1b", ORANGE = "#e8730c", RED = "#d23a2a", BLUE = "#2f6fd6";
const svg = document.getElementById("s");
const rc = rough.svg(svg);
let seed = 7;
const o = (opts = {}) => ({ roughness: 1.3, bowing: 1.2, strokeWidth: 2.2, stroke: INK, seed: seed++, ...opts });
const put = (n) => (svg.appendChild(n), n);

const line = (x1, y1, x2, y2, opts) => put(rc.line(x1, y1, x2, y2, o(opts)));
const rect = (x, y, w, h, opts) => put(rc.rectangle(x, y, w, h, o(opts)));
const ell = (x, y, w, h, opts) => put(rc.ellipse(x, y, w, h, o(opts)));
const path = (d, opts) => put(rc.path(d, o(opts)));
const curve = (pts, opts) => put(rc.curve(pts, o(opts)));
const poly = (pts, opts) => put(rc.polygon(pts, o(opts)));

function text(x, y, s, { size = 30, color = INK, anchor = "start", rotate = 0, weight = 400, font = "Caveat" } = {}) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x); t.setAttribute("y", y);
  t.setAttribute("font-family", `${font}, 'Patrick Hand', cursive`);
  t.setAttribute("font-size", size); t.setAttribute("fill", color);
  t.setAttribute("text-anchor", anchor); t.setAttribute("font-weight", weight);
  if (rotate) t.setAttribute("transform", `rotate(${rotate} ${x} ${y})`);
  t.textContent = s;
  return put(t);
}

// Label: Name / file or symbol / short behaviour, stacked like a whiteboard note.
function label(x, y, name, where, what, { color = INK, anchor = "start", size = 34 } = {}) {
  text(x, y, name, { size, color, anchor, weight: 700 });
  if (where) text(x, y + size * 0.85, where, { size: size * 0.66, color: "#555", anchor, font: "Patrick Hand" });
  if (what) text(x, y + size * 0.85 + size * 0.72, what, { size: size * 0.7, color, anchor });
}

// A short hand-drawn pointer from a label to the thing it names.
const pointer = (x1, y1, x2, y2, color = "#777") =>
  curve([[x1, y1], [(x1 + x2) / 2 + 12, (y1 + y2) / 2 - 10], [x2, y2]], { stroke: color, strokeWidth: 1.4, roughness: 1.6 });

function arrowHead(x, y, angle, color = ORANGE, size = 16) {
  const a1 = angle + 2.6, a2 = angle - 2.6;
  line(x, y, x + size * Math.cos(a1), y + size * Math.sin(a1), { stroke: color, strokeWidth: 2.4 });
  line(x, y, x + size * Math.cos(a2), y + size * Math.sin(a2), { stroke: color, strokeWidth: 2.4 });
}

// Xiaohei: small black deadpan body, two white dot eyes, thin limbs. Arms and legs end at the given absolute points.
function xiaohei(x, y, { s = 1, armL, armR, legL, legR, look = 0 } = {}) {
  const w = 74 * s, h = 92 * s;
  armL = armL || [x - 60 * s, y + 20 * s]; armR = armR || [x + 60 * s, y + 20 * s];
  legL = legL || [x - 22 * s, y + 105 * s]; legR = legR || [x + 22 * s, y + 105 * s];
  line(x - 26 * s, y + 6 * s, armL[0], armL[1], { strokeWidth: 3 * s, roughness: 0.9 });
  line(x + 26 * s, y + 6 * s, armR[0], armR[1], { strokeWidth: 3 * s, roughness: 0.9 });
  line(x - 14 * s, y + 38 * s, legL[0], legL[1], { strokeWidth: 3 * s, roughness: 0.9 });
  line(x + 14 * s, y + 38 * s, legR[0], legR[1], { strokeWidth: 3 * s, roughness: 0.9 });
  ell(x, y, w, h, { fill: INK, fillStyle: "solid", roughness: 0.8 });
  for (const dx of [-13, 13]) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    e.setAttribute("cx", x + (dx + look * 6) * s); e.setAttribute("cy", y - 16 * s);
    e.setAttribute("r", 5.2 * s); e.setAttribute("fill", "#fff");
    put(e);
  }
}

// A page of text: a slightly tilted sheet with scribble lines.
function page(x, y, w = 70, h = 90, { tilt = 0, color = INK, lines = 5 } = {}) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute("transform", `rotate(${tilt} ${x + w / 2} ${y + h / 2})`);
  svg.appendChild(g);
  g.appendChild(rc.rectangle(x, y, w, h, o({ fill: "#fff", fillStyle: "solid", stroke: color })));
  for (let i = 0; i < lines; i++)
    g.appendChild(rc.line(x + 10, y + 16 + i * (h - 26) / lines, x + w - 10 - (i % 2) * 14, y + 16 + i * (h - 26) / lines,
      o({ stroke: "#999", strokeWidth: 1.3, roughness: 1.8 })));
}

// Scissors: two loops and two blades crossing at (x, y).
function scissors(x, y, s = 1, color = INK) {
  ell(x - 26 * s, y + 20 * s, 22 * s, 16 * s, { stroke: color });
  ell(x - 26 * s, y - 20 * s, 22 * s, 16 * s, { stroke: color });
  line(x - 16 * s, y + 14 * s, x + 40 * s, y - 12 * s, { stroke: color, strokeWidth: 2.6 });
  line(x - 16 * s, y - 14 * s, x + 40 * s, y + 12 * s, { stroke: color, strokeWidth: 2.6 });
}
