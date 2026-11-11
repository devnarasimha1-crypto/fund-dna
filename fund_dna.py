import json, re, time, urllib.request
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

AMFI_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
NAV_URL = "https://api.mfapi.in/mf/{}"
MARKET_CODE = 120716  # UTI Nifty 50 Index Fund - Direct Growth, used as the market (Nifty 50) proxy
YEARS, MIN_DAYS, DRIFT_CONF = 3, 600, 0.80
FEATURES = ["ann_vol", "downside_dev", "max_drawdown", "cagr", "ret_1y", "beta", "corr_nifty",
            "skew", "kurt", "neg_day_frac", "autocorr", "worst_day", "best_day"]
HEADER = re.compile(r"Open Ended Schemes\((Equity|Debt|Hybrid|Income/Debt Oriented) Schemes? - (.+)\)")


_AMFI = {"t": 0.0, "text": ""}


def amfi_text(max_age=600):
    """AMFI's daily file of every scheme's newest NAV. Cached for 10 minutes."""
    if time.time() - _AMFI["t"] > max_age or not _AMFI["text"]:
        _AMFI["text"] = urllib.request.urlopen(AMFI_URL, timeout=60).read().decode("utf-8", "ignore")
        _AMFI["t"] = time.time()
    return _AMFI["text"]


_LATEST = {"t": -1.0, "d": {}}


def latest_navs():
    """code -> (date, NAV) from AMFI. This is usually a day fresher than the history API. Parsed once per refresh."""
    text = amfi_text()
    if _LATEST["t"] != _AMFI["t"]:
        rows = [ln.split(";") for ln in text.splitlines() if ln.count(";") == 7]
        df = pd.DataFrame(rows, columns=["code", "a", "b", "name", "plan", "opt", "nav", "date"])
        df["code"] = pd.to_numeric(df["code"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["code", "nav", "date"])
        _LATEST["d"] = {int(c): (d, float(v)) for c, d, v in zip(df["code"], df["date"], df["nav"])}
        _LATEST["t"] = _AMFI["t"]
    return _LATEST["d"]


def list_schemes():
    """All live open-ended Equity/Debt/Hybrid Direct-Growth schemes with their official AMFI category."""
    text = amfi_text()
    rows, cat = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Open Ended Schemes"):
            m = HEADER.match(line)
            cat = ("Debt" if m.group(1) == "Income/Debt Oriented" else m.group(1), m.group(2)) if m else None
        elif cat and line.count(";") == 7:
            code, _, _, name, plan, opt, *_ = line.split(";")
            if "direct" in plan.lower() and "growth" in opt.lower() and "bonus" not in opt.lower():
                rows.append(dict(code=int(code), name=name.strip(), asset_class=cat[0], sebi_category=cat[1]))
    return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)


_NAV = {}


def fetch_nav(code, retries=3, max_age=600):
    """Daily NAV series for the last YEARS years, or None if unavailable. Cached for 10 minutes."""
    hit = _NAV.get(code)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    s = _fetch_nav(code, retries)
    if s is not None:
        _NAV[code] = (time.time(), s)
    return s


def _fetch_nav(code, retries=3):
    for attempt in range(retries):
        try:
            data = json.load(urllib.request.urlopen(NAV_URL.format(code), timeout=30))["data"]
            s = pd.Series([float(x["nav"]) for x in data], index=pd.to_datetime([x["date"] for x in data], format="%d-%m-%Y")).sort_index()
            try:  # add AMFI's newest published NAV when the history API is a day behind
                d, v = latest_navs().get(int(code), (None, None))
                if d is not None and d > s.index[-1] and v > 0:
                    s.loc[d] = v
                    s = s.sort_index()
            except Exception:
                pass
            return s[s.index >= pd.Timestamp.today().normalize() - pd.DateOffset(years=YEARS)]
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def compute_features(nav, mkt_ret):
    """Behaviour fingerprint of a fund from its NAV history. None if history is too short."""
    if nav is None or len(nav) < MIN_DAYS or (nav <= 0).any():
        return None
    r = nav.pct_change().dropna()
    r = r[r.abs() < 0.25]  # drop corporate-action / data-error jumps
    a = pd.concat([r, mkt_ret], axis=1, join="inner").dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    year_ago = nav[nav.index <= nav.index[-1] - pd.Timedelta(days=365)]
    neg = r[r < 0]
    f = dict(
        ann_vol=r.std() * np.sqrt(250),
        downside_dev=neg.std() * np.sqrt(250) if len(neg) > 1 else 0.0,  # 0 when the fund never fell
        max_drawdown=(nav / nav.cummax() - 1).min(),
        cagr=(nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
        ret_1y=nav.iloc[-1] / year_ago.iloc[-1] - 1,
        beta=np.cov(a.iloc[:, 0], a.iloc[:, 1])[0, 1] / a.iloc[:, 1].var(),
        corr_nifty=a.iloc[:, 0].corr(a.iloc[:, 1]),
        skew=r.skew(), kurt=r.kurt(),
        neg_day_frac=(r < 0).mean(),
        autocorr=r.autocorr(),
        worst_day=r.min(), best_day=r.max(),
    )
    return None if any(np.isnan(v) for v in f.values()) else f  # e.g. flat NAV: correlation undefined


class RBFNetwork(BaseEstimator, ClassifierMixin):
    """RBF neural network: Gaussian hidden layer whose centres come from K-Means, softmax output layer."""

    def __init__(self, k=20, width=2.0, C=10.0, class_weight=None, random_state=0):
        self.k, self.width, self.C, self.class_weight, self.random_state = k, width, C, class_weight, random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        self.kmeans_ = KMeans(self.k, n_init=10, random_state=self.random_state).fit(Z)
        self.centers_ = self.kmeans_.cluster_centers_
        d = np.linalg.norm(self.centers_[:, None] - self.centers_[None], axis=2)
        self.sigma_ = self.width * d[d > 0].mean() / np.sqrt(2 * self.k)
        self.output_layer_ = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=10000).fit(self.hidden(X), y)
        self.coef_, self.intercept_ = self.output_layer_.coef_, self.output_layer_.intercept_
        self.classes_ = self.output_layer_.classes_
        return self

    def hidden(self, X):
        Z = (np.asarray(X, float) - self.mean_) / self.scale_
        return np.exp(-((Z[:, None] - self.centers_[None]) ** 2).sum(2) / (2 * self.sigma_ ** 2))

    def predict_proba(self, X):
        s = self.hidden(X) @ self.coef_.T + self.intercept_
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def save(self, path):
        np.savez(path, mean=self.mean_, scale=self.scale_, centers=self.centers_, sigma=self.sigma_,
                 coef=self.coef_, intercept=self.intercept_, classes=self.classes_.astype(str), features=np.array(FEATURES))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(k=len(z["centers"]))
        m.mean_, m.scale_, m.centers_, m.sigma_ = z["mean"], z["scale"], z["centers"], float(z["sigma"])
        m.coef_, m.intercept_, m.classes_ = z["coef"], z["intercept"], z["classes"]
        return m


def verdict(declared, probs):
    pred = max(probs, key=probs.get)
    if pred == declared:
        return "Consistent", f"Behaves like the **{pred}** class, matching its declared category."
    if probs[pred] >= DRIFT_CONF:
        return "Possible style drift", f"Declared **{declared}**, but behaves like **{pred}** ({probs[pred]:.0%} confidence)."
    return "Borderline", f"Declared **{declared}**; behaviour is mixed (closest to **{pred}**, {probs[pred]:.0%})."


def analyse_fund(code, model, schemes):
    """Live end-to-end analysis of one fund: fetch NAVs, build features, predict, compare with label."""
    row = schemes.loc[schemes.code == code].iloc[0]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:  # the fund and the Nifty 50 proxy download side by side
        a, b = ex.submit(fetch_nav, code), ex.submit(fetch_nav, MARKET_CODE)
        nav, mkt = a.result(), b.result()
    if mkt is None:
        raise RuntimeError("Could not reach the NAV data service. Try again in a minute.")
    f = compute_features(nav, mkt.pct_change().dropna())
    if f is None:
        return None
    probs = dict(zip(map(str, model.classes_), model.predict_proba([[f[c] for c in FEATURES]])[0].round(4).tolist()))
    status, text = verdict(row.asset_class, probs)
    return dict(name=row["name"], declared=row.asset_class, sebi_category=row.sebi_category,
                probs=probs, status=status, text=text, features=f, nav=nav, as_of=nav.index[-1].date())
