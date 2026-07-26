"""
Control experiment: can coordinate decoding replace metric learning?

Answer: no. Distances computed from linearly DECODED CAM16-LCD coordinates
reproduce Munsell unit steps only as well as CAM16-LCD itself does
(Group-k STRESS 0.299 for BOTH layers vs 0.288 for true CAM16-LCD coords,
including the full hue failure H~0.50 inherited from CAM16), while the direct
metric map reaches 0.102 (VL) / 0.173 (LM).

This resolves the apparent paradox "LM decodes coordinates better, yet VL
distances agree with humans better":
  (1) different ground truths — coordinate decoding is graded against
      CAM16-LCD (an imperfect analytic model), the metric map against raw
      human data; a perfect CAM16-LCD reproducer is capped at CAM16's own
      misfit with humans;
  (2) global vs local — coordinate regression is a smooth global task,
      nearly saturated for both layers (R2>0.998; 0.55 vs 0.70 dE is
      second-order), whereas unit-step distances probe the local geometry
      (Jacobian) of the embedding, where VL is cleaner;
  (3) differences amplify per-sample noise — LM pooling (over a long token
      sequence with text) adds local noise that cancels in absolute
      positions but doubles in pairwise differences.
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16LCD


def load_module(p):
    s = importlib.util.spec_from_file_location(Path(p).stem, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    mm = load_module("scripts/map_munsell_group_stress.py")
    df, VL, LM = mm.load()
    groups = mm.build_pairs(df)
    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100]))
    Ytrue = XYZ_to_CAM16LCD(XYZ)

    def sub(pr, S):
        return pr[[a in S and b in S for a, b in pr]]

    print("Munsell unit-step STRESS (group-k) via DECODED CAM16-LCD coordinates:")
    res = {}
    for name, X in [("VL", VL), ("LM", LM)]:
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        fold_pg = {g: [] for g in groups}
        for tr_i, te_i in KFold(5, shuffle=True, random_state=42).split(Xs):
            reg = Ridge(alpha=1.0).fit(Xs[tr_i], Ytrue[tr_i])
            Yhat = Ytrue.copy()
            Yhat[te_i] = reg.predict(Xs[te_i])
            te = set(te_i.tolist())
            for g in groups:
                tep = sub(groups[g], te)
                if len(tep) < 3:
                    continue
                d = np.linalg.norm(Yhat[tep[:, 0]] - Yhat[tep[:, 1]], axis=1)
                fold_pg[g].append(mm.stress(d, np.ones(len(d))))
        means = {g: np.mean(v) for g, v in fold_pg.items()}
        print(f"  {name}: H={means['varying-H']:.3f} C={means['varying-C']:.3f} "
              f"V={means['varying-V']:.3f}  -> Group-k mean = "
              f"{np.mean(list(means.values())):.3f}")
        res[name] = {"group_k_mean": round(float(np.mean(list(means.values()))), 4),
                     "per_group": {g: round(float(v), 4) for g, v in means.items()}}
    print("\nreference: true CAM16-LCD coords Group-k = 0.288 (H 0.504/C 0.286/V 0.074)")
    print("           direct metric map: VL 0.102 / LM 0.173")

    import json
    from datetime import date
    out = Path("data/analysis"); out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "script": "scripts/decoded_coords_control.py",
        "date": str(date.today()),
        "protocol": "контроль: расстояния через ДЕКОДИРОВАННЫЕ координаты CAM16-LCD "
                    "(Ridge, 5-фолд по цветам), group-k STRESS на единичных шагах",
        "results": res,
    }, open(out / "decoded_coords_control.json", "w"), indent=2, ensure_ascii=False)
    print("saved data/analysis/decoded_coords_control.json")


if __name__ == "__main__":
    main()
