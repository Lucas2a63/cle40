"""Vérifie que le bras nouveau et les mesures nouvelles de la clé 43 tournent : deux générations d'INDEP0 et
d'INDEP (avec mesures finales) sur la graine 201, hors plan. N'affiche que les durées et la forme des sorties."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelle43 as E

for bras, fin in (("INDEP0", False), ("INDEP", True)):
    t0 = time.time()
    r = E.course(bras, "mlp", 201, gens=2, finales=fin, res=32)
    dt = time.time() - t0
    j = r["juge"][-1]
    print(bras, f"{dt / 2:.1f} s/gén (mesures finales comprises)" if fin else f"{dt / 2:.1f} s/gén",
          "clés du juge", len(j), "note_ens" in j,
          "fin_ens" in r and sorted(r["fin_ens"].keys()),
          "front" in r.get("fin_ens", {}) and len(r["fin_ens"]["front"]["Rf"]),
          "ancre" in r.get("fin_ens", {}) and (r["fin_ens"]["ancre"]["n"], r["fin_ens"]["ancre"]["chats"]), flush=True)
