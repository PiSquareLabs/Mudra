// Service worker registration, install prompt, copy buttons and a small input-format checker.
if ("serviceWorker" in navigator) {
  addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}

let deferred = null;
const installBtn = document.getElementById("install");
addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferred = e;
  installBtn.hidden = false;
});
installBtn.addEventListener("click", async () => {
  if (!deferred) return;
  deferred.prompt();
  await deferred.userChoice;
  deferred = null;
  installBtn.hidden = true;
});
addEventListener("appinstalled", () => { installBtn.hidden = true; });

document.querySelectorAll("pre[data-copy]").forEach((pre) => {
  const b = document.createElement("button");
  b.className = "copy";
  b.textContent = "Copy";
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.innerText.replace(/^Copy\n?/, ""));
      b.textContent = "Copied";
    } catch { b.textContent = "Select + copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  });
  pre.appendChild(b);
});

// Input-format checker: paste JSON of shape [32][61][2] (null = missing point).
const T = 32, N = 61;
document.getElementById("check").addEventListener("click", () => {
  const out = document.getElementById("result");
  const fail = (m) => { out.className = "bad"; out.textContent = m; };
  let a;
  try { a = JSON.parse(document.getElementById("json").value); } catch { return fail("Not valid JSON."); }
  if (!Array.isArray(a) || a.length !== T) return fail(`Expected ${T} frames, got ${Array.isArray(a) ? a.length : "a non-array"}.`);
  let missing = 0, hands = [0, 0];
  for (let t = 0; t < T; t++) {
    const f = a[t];
    if (!Array.isArray(f) || f.length !== N) return fail(`Frame ${t}: expected ${N} points.`);
    for (let p = 0; p < N; p++) {
      const pt = f[p];
      if (pt === null) { missing++; continue; }
      if (!Array.isArray(pt) || pt.length !== 2) return fail(`Frame ${t}, point ${p}: expected [x, y] or null.`);
      const [x, y] = pt;
      if (x === null && y === null) { missing++; continue; }
      if (typeof x !== "number" || typeof y !== "number" || x < -0.5 || x > 1.5 || y < -0.5 || y > 1.5)
        return fail(`Frame ${t}, point ${p}: x/y must be numbers in roughly 0–1 image coordinates.`);
      if (p === 19) hands[0]++;
      if (p === 40) hands[1]++;
    }
  }
  out.className = "ok";
  out.textContent = `OK: [1, ${T}, ${N}, 2]. ${missing} missing points (send NaN). Left hand in ${hands[0]} frames, right hand in ${hands[1]} frames.`;
});
document.getElementById("sample").addEventListener("click", () => {
  const frame = Array.from({ length: N }, (_, p) => (p >= 40 ? null : [0.5, 0.5]));
  document.getElementById("json").value = JSON.stringify(Array.from({ length: T }, () => frame));
});
