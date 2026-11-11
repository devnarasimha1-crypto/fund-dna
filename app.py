import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import fund_dna as fd

HERE = Path(__file__).parent
model = fd.RBFNetwork.load(HERE / "fund_dna_model.npz")
RANGES = json.loads((HERE / "ranges.json").read_text())
META = json.loads((HERE / "meta.json").read_text())
schemes = fd.list_schemes()
FUNDS = [{"code": int(r.code), "name": r.name, "category": f"{r.asset_class} - {r.sebi_category}"}
         for r in schemes.sort_values("name").itertuples()]
CODES = {f["code"] for f in FUNDS}

app = FastAPI(title="Fund DNA")


@app.get("/api/meta")
def meta():
    return META


@app.get("/api/funds")
def funds():
    return FUNDS


@app.get("/api/analyse/{code}")
def analyse(code: int):
    if code not in CODES:
        raise HTTPException(404, "Unknown scheme code.")
    try:
        res = fd.analyse_fund(code, model, schemes)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if res is None:
        raise HTTPException(422, "short_history")
    return {
        "name": res["name"], "declared": res["declared"], "sebi_category": res["sebi_category"],
        "status": res["status"], "probs": res["probs"], "predicted": max(res["probs"], key=res["probs"].get),
        "features": {k: float(v) for k, v in res["features"].items()}, "ranges": RANGES,
        "nav": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in res["nav"].items()],
        "as_of": str(res["as_of"]),
    }


# The React build is unpacked into ./static by the notebook; mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="site")
