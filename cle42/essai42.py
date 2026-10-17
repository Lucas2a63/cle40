"""Les nouveaux bras tournent (deux générations, sans mesures finales), sur la graine 96, hors plan :
python3 essai42.py. N'affiche que la durée, le calcul et la forme des sorties."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelle42 as E
for bras in ("SYSL", "SYSR"):
    t = time.time()
    r = E.course(bras, "mlp", 96, gens=2, finales=False, res=32)
    j = r["juge"][-1]
    print(bras, f"{(time.time() - t) / 2:.1f} s/gén", "calcul", r["calcul"], "génome", sorted(r["genome"]),
          "juge g", j["g"], "clés", len(j), flush=True)
