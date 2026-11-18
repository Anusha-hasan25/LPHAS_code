"""LPHAS reference implementation (NumPy only): synthetic telemetry, LSTM forecaster with BPTT,
nonparametric dynamic thresholding (NDT), event-wise evaluation, post-training weight quantization."""
import numpy as np

C = 8
PERIODS = np.array([90, 120, 150, 200, 90, 300, 240, 180], float)

# ---------------------------------------------------------------- telemetry
def gen_series(T, seed, anom_ranges=None, storm_pulses=12):
    r = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    A = r.uniform(1.0, 2.0, C); ph = r.uniform(0, 2*np.pi, C); ph2 = r.uniform(0, 2*np.pi, C)
    base = np.stack([A[c]*np.sin(2*np.pi*t/PERIODS[c]+ph[c]) + 0.4*A[c]*np.sin(2*np.pi*t/(PERIODS[c]/3)+ph2[c])
                     for c in range(C)], 1)
    base[:, 4] += 0.6*base[:, 0]; base[:, 5] -= 0.5*base[:, 1]
    storm = np.ones(T); offs = np.zeros(T)
    for _ in range(storm_pulses):
        c0 = r.integers(1500, T-1500); w = r.integers(150, 400)
        g = np.exp(-0.5*((t-c0)/w)**2); storm += 2.0*g; offs += 0.5*g
    sig = r.uniform(0.05, 0.12, C)
    noise = np.zeros((T, C)); e = r.standard_normal((T, C))
    for i in range(1, T): noise[i] = 0.8*noise[i-1] + sig*storm[i]*e[i]
    X = base + noise; X[:, 2] += offs; X[:, 3] += 0.8*offs
    labels = np.zeros(T, int); events = []
    for (lo, hi, n) in (anom_ranges or []):
        types = ["spike", "level", "drift", "freq", "stuck", "noise"]
        starts = np.sort(r.choice(np.arange(lo, hi, 300), size=n, replace=False))
        for k, s0 in enumerate(starts):
            L = int(r.integers(25, 70)); ch = int(r.integers(0, C)); ty = types[k % len(types)]
            idx = np.arange(s0, s0+L)
            if ty == "spike": X[idx[::3], ch] += r.choice([-1, 1])*A[ch]*r.uniform(0.8, 1.2)
            elif ty == "level": X[idx, ch] += 0.6*A[ch]*r.choice([-1, 1])
            elif ty == "drift": X[idx, ch] += np.linspace(0, 0.9*A[ch], L)
            elif ty == "freq": X[idx, ch] += A[ch]*(np.sin(2*np.pi*idx/(0.6*PERIODS[ch])+ph[ch]) - np.sin(2*np.pi*idx/PERIODS[ch]+ph[ch]))
            elif ty == "stuck": X[idx, ch] = X[idx[0], ch]
            elif ty == "noise": X[idx, ch] += 4*sig[ch]*storm[idx]*r.standard_normal(L)
            labels[idx] = 1; events.append((int(s0), int(s0+L), ty, ch))
    return X, labels, events, storm

# ---------------------------------------------------------------- LSTM
def sig_(x): return 1/(1+np.exp(-np.clip(x, -30, 30)))

class LSTM:
    def __init__(self, H=32, seed=0):
        r = np.random.default_rng(seed); self.H = H
        s = 1/np.sqrt(H)
        self.p = {"Wx": r.uniform(-s, s, (C, 4*H)), "Wh": r.uniform(-s, s, (H, 4*H)),
                  "b": np.zeros(4*H), "Wo": r.uniform(-s, s, (H, C)), "bo": np.zeros(C)}
        self.p["b"][H:2*H] = 1.0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0
    def n_params(self): return int(sum(v.size for v in self.p.values()))
    def forward(self, X, keep=False):
        B, T, _ = X.shape; H = self.H; p = self.p
        h = np.zeros((B, H)); c = np.zeros((B, H)); Y = np.zeros((B, T, C)); cache = []
        for t in range(T):
            z = X[:, t] @ p["Wx"] + h @ p["Wh"] + p["b"]
            i, f, g, o = sig_(z[:, :H]), sig_(z[:, H:2*H]), np.tanh(z[:, 2*H:3*H]), sig_(z[:, 3*H:])
            cn = f*c + i*g; tc = np.tanh(cn); hn = o*tc
            if keep: cache.append((X[:, t], h, c, i, f, g, o, cn, tc))
            h, c = hn, cn; Y[:, t] = h @ p["Wo"] + p["bo"]
            if keep: cache[-1] = cache[-1] + (h,)
        return (Y, cache) if keep else Y
    def loss_grad(self, X, Yt):
        Y, cache = self.forward(X, keep=True); B, T, _ = X.shape; H = self.H; p = self.p
        d = Y - Yt; loss = float((d**2).mean()); dY = 2*d/d.size
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_n = np.zeros((B, H)); dc_n = np.zeros((B, H))
        for t in reversed(range(T)):
            x, hp, cp, i, f, gg, o, cn, tc, hn = cache[t]
            g["Wo"] += hn.T @ dY[:, t]; g["bo"] += dY[:, t].sum(0)
            dh = dY[:, t] @ p["Wo"].T + dh_n
            do = dh*tc; dc = dh*o*(1-tc**2) + dc_n
            di = dc*gg; dg = dc*i; df = dc*cp; dc_n = dc*f
            dz = np.concatenate([di*i*(1-i), df*f*(1-f), dg*(1-gg**2), do*o*(1-o)], 1)
            g["Wx"] += x.T @ dz; g["Wh"] += hp.T @ dz; g["b"] += dz.sum(0); dh_n = dz @ p["Wh"].T
        return loss, g
    def step(self, g, lr=3e-3, clip=1.0, b1=0.9, b2=0.999):
        n = np.sqrt(sum((v**2).sum() for v in g.values()))
        if n > clip: g = {k: v*clip/n for k, v in g.items()}
        self.t += 1
        for k in self.p:
            self.m[k] = b1*self.m[k]+(1-b1)*g[k]; self.v[k] = b2*self.v[k]+(1-b2)*g[k]**2
            self.p[k] -= lr*(self.m[k]/(1-b1**self.t))/(np.sqrt(self.v[k]/(1-b2**self.t))+1e-8)

def make_windows(Z, W=64, stride=8):
    idx = np.arange(0, len(Z)-W-1, stride)
    return np.stack([Z[i:i+W+1] for i in idx])

def train(model, Ztr, Zva, epochs=40, B=64, lr=3e-3, seed=0, W=64, stride=8, log=None):
    r = np.random.default_rng(seed); Wtr = make_windows(Ztr, W, stride); Wva = make_windows(Zva, W, 32)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        perm = r.permutation(len(Wtr)); tl = []
        cur = lr*(0.5*(1+np.cos(np.pi*ep/epochs))*0.9+0.1)
        for s in range(0, len(perm), B):
            b = Wtr[perm[s:s+B]]; l, g = model.loss_grad(b[:, :-1], b[:, 1:]); model.step(g, lr=cur); tl.append(l)
        vl = float(((model.forward(Wva[:, :-1])-Wva[:, 1:])**2).mean())
        hist["train"].append(float(np.mean(tl))); hist["val"].append(vl)
        if log and ep % 10 == 0: log(f"  ep {ep} train {hist['train'][-1]:.4f} val {vl:.4f}")
    return hist

def predict_stream(model, Z, W=64, warm=32):
    """one-step-ahead prediction for every step via overlapping windows (state reset each window)."""
    T = len(Z); step = W-warm; starts = np.arange(0, T-W-1, step)
    Xw = np.stack([Z[s:s+W] for s in starts]); Y = model.forward(Xw)
    pred = np.full((T, C), np.nan)   # pred[t] forecasts Z[t]
    for k, s in enumerate(starts):
        lo = 0 if k == 0 else warm
        pred[s+1+lo:s+W+1] = Y[k, lo:]
    return pred

# ---------------------------------------------------------------- NDT & evaluation
def ewma(x, span=20):
    a = 2/(span+1); out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a*x[i]+(1-a)*out[i-1]
    return out

def score_series(Z, pred, sd_val):
    err = np.abs(Z-pred)/sd_val
    err = np.nan_to_num(err, nan=0.0)
    return np.max(np.stack([ewma(err[:, c]) for c in range(C)], 1), 1)

def sequences(mask, gap=5):
    idx = np.where(mask)[0]
    if len(idx) == 0: return []
    seqs = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i-p > gap: seqs.append((s, p+1)); s = i
        p = i
    seqs.append((s, p+1)); return seqs

def ndt(s, seg=1500, p_prune=0.13, zs=np.arange(2.5, 12.1, 0.5)):
    flag = np.zeros(len(s), bool)
    for a in range(0, len(s), seg):
        e = s[a:a+seg]
        if len(e) < 50: continue
        mu, sd = e.mean(), e.std(); best = None
        for z in zs:
            eps = mu+z*sd; below = e[e < eps]
            if len(below) < 10: continue
            above = e >= eps; ea = above.sum(); nseq = len(sequences(above))
            if ea == 0: continue
            sc = ((mu-below.mean())/mu + (sd-below.std())/sd)/(ea+nseq**2)
            if best is None or sc > best[0]: best = (sc, eps)
        if best is None: continue
        eps = best[1]; above = e >= eps; seqs = sequences(above)
        if not seqs: continue
        mx = np.array([e[x:y].max() for x, y in seqs]); order = np.argsort(-mx)
        m_below = e[e < eps].max() if (e < eps).any() else 0
        ms = list(mx[order])+[m_below]; keep_n = 0
        for i in range(len(seqs)):
            if (ms[i]-ms[i+1])/ms[i] > p_prune: keep_n = i+1
        for j in order[:keep_n]:
            x, y = seqs[j]; flag[a+x:a+y] = True
    return flag

def event_eval(flag, events, gap=5, tol=10):
    seqs = sequences(flag, gap); matched_true = set(); tp_seq = 0
    for (x, y) in seqs:
        hit = False
        for k, ev in enumerate(events):
            if x < ev[1]+tol and y > ev[0]-tol: hit = True; matched_true.add(k)
        tp_seq += hit
    fp = len(seqs)-tp_seq; tp = len(matched_true); fn = len(events)-tp
    pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return dict(precision=pr, recall=rc, f1=f1, fp=int(fp), tp=int(tp), fn=int(fn))

# ---------------------------------------------------------------- quantisation
def quantize_weights(model, bits=8):
    q = {}; qmax = 2**(bits-1)-1; size = 0
    for k, v in model.p.items():
        if k in ("b", "bo"): q[k] = v.copy(); size += v.size*4; continue
        sc = np.abs(v).max()/qmax; q[k] = np.round(v/sc).clip(-qmax-1, qmax)*sc; size += v.size*bits/8+4
    m2 = LSTM(model.H); m2.p = q; return m2, size
