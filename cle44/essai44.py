"""Vérifie que les bras nouveaux et les mesures nouvelles de la clé 44 tournent : deux générations d'ENS (avec mesures
finales) et d'ENSR sur la graine 201, hors plan. N'affiche que les durées et la forme des sorties."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelle44 as E

for bras, fin in (("ENSR", False), ("ENS", True)):
    t0 = time.time()
    r = E.course(bras, "mlp", 201, gens=2, finales=fin, res=32)
    dt = time.time() - t0
    j = r["juge"][-1]
    print(bras, f"{dt / 2:.1f} s/gén (mesures finales comprises)" if fin else f"{dt / 2:.1f} s/gén",
          "clés du juge", len(j), sorted(k for k in j if k in ("amb", "amb_chat", "desacc", "brier_moy", "brier_ens")),
          "clés de tr", sorted(k for k in r["tr"][-1] if k.startswith(("apport", "lv_", "llr_sd", "lwd_sd"))),
          "lignées", j.get("lignees"),
          "fin_ens" in r and sorted(r["fin_ens"].get("diversite", {}).keys()), flush=True)
