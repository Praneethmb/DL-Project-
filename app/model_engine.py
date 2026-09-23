import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

rng = np.random.default_rng(42)

# ---------- Data Simulators ----------
def sim_wildfire(n=1200):
    cols = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
    d = np.column_stack([
        rng.normal(32,5,n), rng.normal(60,18,n).clip(10,100),
        rng.normal(15,6,n).clip(2,40), rng.gamma(1,1.2,n).clip(0,20),
        rng.normal(80,12,n).clip(30,99), rng.normal(20,12,n).clip(1,70),
        rng.normal(90,45,n).clip(5,250), rng.normal(6,4,n).clip(0,25),
        rng.normal(25,14,n).clip(1,80), rng.normal(9,7,n).clip(0,40)
    ])
    ds = (d - d.mean(0)) / d.std(0)
    w = np.array([0.8, -1.1, 0.4, -0.7, 0.9, 0.3, 0.2, 1.0, 0.3, 1.4])
    y = (ds @ w + rng.normal(0,0.6,n) > np.median(ds @ w)).astype(int)
    return pd.DataFrame(d, columns=cols), y, d.mean(0), d.std(0)

def sim_flood(n=1400):
    cols = ["MonsoonIntensity","TopographyDrainage","RiverManagement","Deforestation","Urbanization",
            "ClimateChange","DamsQuality","Siltation","DrainageSystems","Rainfall"]
    d = np.column_stack([rng.normal(5,2,n).clip(0,10) for _ in range(9)] + [rng.normal(120,40,n).clip(10,300)])
    ds = (d - d.mean(0)) / d.std(0)
    w = np.array([1.2, -0.4, -0.5, 0.7, 0.6, 0.5, -0.8, 0.6, -1.0, 1.3])
    y = (ds @ w + rng.normal(0,0.6,n) > np.median(ds @ w)).astype(int)
    return pd.DataFrame(d, columns=cols), y, d.mean(0), d.std(0)

# ---------- Feature Attention NN ----------
class FeatureAttentionNN:
    def __init__(self, n_features, hidden=32, lr=0.08, epochs=600, seed=1):
        r = np.random.default_rng(seed)
        self.F = n_features
        self.theta = np.zeros(n_features)
        self.W1 = r.normal(0, np.sqrt(2/n_features), (n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, np.sqrt(2/hidden), (hidden, 1))
        self.b2 = np.zeros(1)
        self.lr, self.epochs = lr, epochs

    def attention(self):
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    def forward(self, X):
        a = self.attention()
        s = a * self.F
        Xw = X * s
        z1 = Xw @ self.W1 + self.b1
        h = np.maximum(0, z1)
        p = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))
        return p.ravel(), (X, a, Xw, z1, h)

    def fit(self, X, y):
        n = len(X)
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):
            p, (X_, a, Xw, z1, h) = self.forward(X)
            p = p.reshape(-1, 1)
            dz2 = (p - y) / n
            dW2 = h.T @ dz2
            db2 = dz2.sum(0)
            dz1 = (dz2 @ self.W2.T) * (z1 > 0)
            dW1 = Xw.T @ dz1
            db1 = dz1.sum(0)
            ds = ((dz1 @ self.W1.T) * X_).sum(0)
            da = ds * self.F
            dtheta = a * (da - (a * da).sum())
            for pr, g in [(self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2), (self.theta, dtheta)]:
                pr -= self.lr * g
        return self

    def predict_proba(self, X):
        return self.forward(X)[0]

# ---------- Minimal LSTM ----------
class LSTM:
    def __init__(self, F, H=10, seed=0):
        r = np.random.default_rng(seed)
        s = 1 / np.sqrt(H)
        self.H = H
        self.W = r.normal(0, s, (4*H, F+H))
        self.b = np.zeros(4*H)
        self.b[H:2*H] = 1.0
        self.Wy = r.normal(0, s, (1, H))
        self.by = np.zeros(1)
        self.params = ["W", "b", "Wy", "by"]
        self.m = {p:0 for p in self.params}
        self.v = {p:0 for p in self.params}
        self.t = 0

    def sig(self, x): return 1 / (1 + np.exp(-x))

    def forward(self, X):
        N, T, F = X.shape
        H = self.H
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        self.cache = []
        for t in range(T):
            z = np.concatenate([X[:,t,:], h], 1)
            g = z @ self.W.T + self.b
            i = self.sig(g[:,:H])
            f = self.sig(g[:, H:2*H])
            o = self.sig(g[:, 2*H:3*H])
            u = np.tanh(g[:, 3*H:])
            c = f * c + i * u
            tc = np.tanh(c)
            h = o * tc
            self.cache.append((z, i, f, o, u, c, tc, h))
        self.h_last = h
        return self.sig(h @ self.Wy.T + self.by).ravel()

    def backward(self, X, y, p):
        N, T, F = X.shape
        H = self.H
        dWy = (((p-y)[:,None]*self.h_last).sum(0, keepdims=True)) / N
        dby = np.array([(p-y).mean()])
        dh = ((p-y)[:,None] @ self.Wy) / N
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dc = np.zeros((N, H))
        for t in reversed(range(T)):
            z, i, f, o, u, c, tc, h = self.cache[t]
            c_prev = self.cache[t-1][5] if t > 0 else np.zeros((N, H))
            do = dh * tc
            dc = dc + dh * o * (1 - tc**2)
            di = dc * u
            du = dc * i
            df = dc * c_prev
            di *= i * (1 - i)
            df *= f * (1 - f)
            do *= o * (1 - o)
            du *= (1 - u**2)
            dg = np.concatenate([di, df, do, du], 1)
            dW += dg.T @ z / N
            db += dg.mean(0)
            dz = dg @ self.W
            dh = dz[:, F:]
            dc = dc * f
        for g in (dW, dWy): np.clip(g, -5, 5, out=g)
        return {"W": dW, "b": db, "Wy": dWy, "by": dby}

    def step(self, gr, lr, wd=1e-3):
        self.t += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        for p in self.params:
            g = gr[p] + (wd * getattr(self, p) if p in ("W", "Wy") else 0)
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g * g
            setattr(self, p, getattr(self, p) - lr * (self.m[p] / (1 - b1**self.t)) / (np.sqrt(self.v[p] / (1 - b2**self.t)) + e))

    def fit(self, X, y, epochs=30, lr=0.015, batch=64, seed=0):
        rr = np.random.default_rng(seed)
        for _ in range(epochs):
            idx = rr.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s+batch]
                p = np.clip(self.forward(X[b]), 1e-7, 1-1e-7)
                self.step(self.backward(X[b], y[b], p), lr)
        return self

# Global Engine Container
class PredictionEngine:
    def __init__(self):
        # 1. Wildfire
        Xw, yw, self.w_mean, self.w_std = sim_wildfire()
        self.w_sc = StandardScaler().fit(Xw.values)
        self.fann_w = FeatureAttentionNN(Xw.shape[1]).fit(self.w_sc.transform(Xw.values), yw)
        self.w_cols = list(Xw.columns)
        
        # 2. Flood
        Xf, yf, self.f_mean, self.f_std = sim_flood()
        self.f_sc = StandardScaler().fit(Xf.values)
        self.fann_f = FeatureAttentionNN(Xf.shape[1]).fit(self.f_sc.transform(Xf.values), yf)
        self.f_cols = list(Xf.columns)
        
        # 3. Time Series LSTM
        def make_series(days=2000):
            def region(nd, seed):
                r = np.random.default_rng(seed)
                t = np.arange(nd)
                season = np.sin((t/nd)*2*np.pi*2)
                temp, rh, ws = np.zeros(nd), np.zeros(nd), np.zeros(nd)
                temp[0], rh[0], ws[0] = 30, 55, 15
                for i in range(1, nd):
                    temp[i] = 0.7*temp[i-1] + 0.3*(28+7*season[i]) + r.normal(0,1.6)
                    rh[i] = 0.7*rh[i-1] + 0.3*(60-22*season[i]) + r.normal(0,4)
                    ws[i] = 0.6*ws[i-1] + 0.4*15 + r.normal(0,3)
                rh = rh.clip(10,100); ws = ws.clip(2,40)
                rain = np.where(r.random(nd)<0.22, r.gamma(2,3,nd), 0.0).clip(0,30)
                D = np.zeros(nd)
                for i in range(1, nd): D[i] = 0.85*D[i-1] + (1.0 if rain[i]<3 else -4.0)
                Dn = (D - D.mean()) / (D.std() + 1e-9)
                risk = 2.6*Dn + 0.15*(temp - temp.mean()) / temp.std()
                y = (r.random(nd) < 1/(1+np.exp(-risk))).astype(int)
                return np.column_stack([temp, rh, ws, rain]), y
            Xa, ya = region(days//2, 1)
            Xb, yb = region(days//2, 2)
            return np.vstack([Xa, Xb]), np.concatenate([ya, yb])
            
        Xs, ys = make_series()
        self.s_mean, self.s_std = Xs.mean(0), np.where(Xs.std(0)==0, 1, Xs.std(0))
        Xs_norm = (Xs - self.s_mean) / self.s_std
        
        K = 12
        xs = [Xs_norm[i:i+K] for i in range(len(Xs_norm)-K)]
        ys_win = [ys[i+K] for i in range(len(Xs_norm)-K)]
        XW, yW = np.array(xs), np.array(ys_win)
        self.lstm = LSTM(4, H=10).fit(XW[:1200], yW[:1200], epochs=25)

    def predict_wildfire(self, feat_dict):
        arr = np.array([[feat_dict[c] for c in self.w_cols]])
        arr_s = self.w_sc.transform(arr)
        prob = float(self.fann_w.predict_proba(arr_s)[0])
        att = {col: float(val) for col, val in zip(self.w_cols, self.fann_w.attention())}
        return prob, att

    def predict_flood(self, feat_dict):
        arr = np.array([[feat_dict[c] for c in self.f_cols]])
        arr_s = self.f_sc.transform(arr)
        prob = float(self.fann_f.predict_proba(arr_s)[0])
        att = {col: float(val) for col, val in zip(self.f_cols, self.fann_f.attention())}
        return prob, att

    def predict_sequence(self, dry_spell_days=10, init_temp=35.0):
        # Generate a synthetic 12-day sequence based on dry_spell_days
        days = 12
        temps = np.linspace(init_temp - 4, init_temp, days)
        rhs = np.linspace(65 - dry_spell_days*3, 30 - dry_spell_days*2, days).clip(15, 90)
        wss = np.linspace(10, 22, days)
        rains = np.zeros(days)
        if dry_spell_days < 5:
            rains[0] = 12.0
            rains[1] = 5.0

        raw_seq = np.column_stack([temps, rhs, wss, rains])
        seq_norm = (raw_seq - self.s_mean) / self.s_std
        
        daily_probs = []
        for i in range(1, days + 1):
            sub_seq = seq_norm[:i]
            if len(sub_seq) < 12:
                # pad with first day
                pad = np.tile(sub_seq[0], (12 - len(sub_seq), 1))
                sub_seq = np.vstack([pad, sub_seq])
            prob = float(self.lstm.forward(sub_seq[None, :, :])[0])
            daily_probs.append(prob)
            
        return {
            "days": [f"Day {i+1}" for i in range(days)],
            "probabilities": [round(p * 100, 1) for p in daily_probs],
            "temperatures": [round(t, 1) for t in temps],
            "humidity": [round(h, 1) for h in rhs],
            "rainfall": [round(r, 1) for r in rains]
        }

engine = PredictionEngine()
