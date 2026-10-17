"""Les nouveaux bras tournent (deux générations, sans mesures finales) : python3 essai41.py"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelle41 as E
for bras in ("POIDS", "REGLAGES", "SANSCUR", "M32", "COURT2", "SEUL2", "GRAND2"):
    t = time.time()
    r = E.course(bras, "mlp", 98, gens=2, finales=False, lr0=E.LR0 / 8, wd=1e-3, res=32)
    j = r["juge"][-1]
    print(bras, f"{(time.time() - t) / 2:.1f} s/gén", "calcul", r["calcul"], "juge g", j["g"], "clés", len(j), flush=True)
