"""Clé 46 — les courses du plan, avec des points de reprise.

Écrit le 22 octobre 2026 vers 7 h, pendant la campagne, parce que les conteneurs des sessions cloud (et celui du bac à
sable) redémarrent souvent cette nuit : chaque redémarrage tue les processus et fait recommencer de zéro les courses
en cours, dont celles de la recherche aléatoire convolutive (≈ 5 h chacune).

Ce module ne change rien au calcul. course() fait exactement ce que fait echelle46.course pour les bras du plan de la
clé 46 (INDEPC, SEUL10, SEUL, GRAND, COURT) : les corps de course_pop et de course_seul sont recopiés tels quels
depuis echelle46.py (le moteur, inchangé, dont ils appellent toutes les fonctions), et la seule chose ajoutée est, à
la fin de chaque génération (et après chaque étape des mesures finales), l'écriture de tout l'état de la course dans un
fichier (pickle : les générateurs aléatoires, la population, l'état du monde, le tampon, les ancres, le suivi). Si le
fichier existe au départ, la course reprend à la génération suivante ; le résultat est le même, au bit près, que
celui d'une course faite d'une traite (vérifié avant usage : reprise46_essai.py).
"""
import math, os, pickle, time
import numpy as np
import echelle46 as E


def _ecrire(chemin, obj):
    tmp = chemin + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, chemin)


def _lire(chemin):
    if chemin and os.path.exists(chemin):
        with open(chemin, "rb") as f:
            return pickle.load(f)
    return None


def _etat_monde(W):
    return dict(rng=W.rng, deja=W.deja, recycle=W.recycle)


def _monde(seed, st):
    W = E.Monde(seed, "fixe")
    if st is not None:                                   # les données et le découpage sont refaits à l'identique
        W.rng, W.deja, W.recycle = st["rng"], st["deja"], st["recycle"]
    return W


class Arret(Exception):
    """Pour les essais : simule un redémarrage juste après l'écriture du point de la génération donnée."""


def course_pop_r(arch, seed, g_bas, selection=True, gens=E.GENS, finales=True, quoi="tout", sans_curation=False,
                 genome=None, critere="note", choix=False, point=None, arret=None):
    """echelle46.course_pop, avec points de reprise (point : le fichier ; arret : génération après laquelle s'arrêter,
    pour les essais)."""
    st = _lire(point)
    if st is None:
        W = E.Monde(seed, "fixe")
        rng = np.random.default_rng(1000 + seed)
        if g_bas:
            pop = E.Population(rng, arch, "BAS", E.C_ANCRE, g_bas, genome=genome)
        else:
            pop = E.Population(rng, arch, "IMP", E.C_ANCRE, genome=genome)
        ancres = []
        if sans_curation:
            pop.sans_curation = True
            pop.eps[:] = 0.0
        TX, Ty = W.arrivees(E.TAMPON, 0)
        anc = np.arange(E.M)
        tr, jg = [], []
        g0, best, fin = 0, None, {}
    else:
        rng, pop = st["rng"], st["pop"]
        assert pop.rng is rng
        W = _monde(seed, st["W"])
        ancres, TX, Ty, anc, tr, jg, best = st["ancres"], st["TX"], st["Ty"], st["anc"], st["tr"], st["jg"], st["best"]
        g0, fin = st["g"] + 1, st.get("fin", {})
    pj = E.juges(gens, g_bas)

    def sauver(g, fin):
        if point:
            _ecrire(point, dict(g=g, rng=rng, pop=pop, W=_etat_monde(W), ancres=ancres, TX=TX, Ty=Ty, anc=anc,
                                tr=tr, jg=jg, best=best, fin=fin))
        if arret is not None and g == arret and not fin:
            raise Arret(g)

    for g in range(g0, gens):
        pop.regler(g)
        if g > 0:                                         # le monde envoie du neuf (FIFO)
            nx, ny = W.arrivees(E.ARRIVEES, g)
            TX = np.concatenate([TX[E.ARRIVEES:], nx]); Ty = np.concatenate([Ty[E.ARRIVEES:], ny])
            pop.u = np.concatenate([pop.u[:, E.ARRIVEES:],
                                    np.repeat(pop.u.mean(1, keepdims=True), E.ARRIVEES, 1)], 1)
        for _ in range(E.PAS):
            idx = pop.tirer(E.NB)
            pop.pas(TX[idx], Ty[idx], idx)
        FX, Fy, iw = W.test_vise(g, E.C_ANCRE)            # l'ancre
        if g >= gens - E.DERNIERES:
            ancres.append((FX, Fy, iw))
        note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean()), llr_b=float(pop.llr[best]), lwd_b=float(pop.lwd[best]),
                       eps_b=float(pop.eps[best]), livre=best))
        bas, haut = o[:E.NREMP], o[-E.NREMP:]             # la sélection : selon la note propre…
        if critere == "ensemble":                         # … ou (clé 44) selon l'apport à l'ensemble
            apport, lv = E.contributions(pop, FX, Fy, iw)
            oc = np.argsort(apport, kind="stable")
            haut = oc[-E.NREMP:]
            bas = oc[oc != best][:E.NREMP]
            tr[-1].update(lv_ens=lv, apport_min=float(apport.min()), apport_max=float(apport.max()),
                          apport_livre=float(apport[best]), livre_protege=int(best in oc[:E.NREMP]),
                          llr_sd=float(pop.llr.std()), lwd_sd=float(pop.lwd.std()))
        if g in pj:                                       # le juge, que le système ne voit jamais
            d = E.juge_pop(pop, W, g, note, best, ensemble=True)
            d["lignees"] = int(len(np.unique(anc)))
            d["note_ens"] = E.note_ensemble(pop, FX, Fy, iw)
            jg.append(d)
        if selection:                                     # la sélection (SYS, et les bras qui en dérivent)
            src = rng.choice(haut, E.NREMP, replace=False)
            pop.copier(bas, src, quoi)
            if quoi in ("tout", "poids"):
                anc[bas] = anc[src]
        sauver(g, {})
    out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=best,
               calcul=dict(reseaux=E.M, pas_par_generation=E.M * E.PAS, pas=E.M * E.PAS * gens, res=E.RES))
    if genome is not None:
        out["genome"] = {k: [float(x) for x in v] for k, v in genome.items()}
    if finales:
        etapes = [("fin", lambda: E.mesures_finales(pop, W)),
                  ("fin_livre", lambda: E.finales_reseau(pop, W, best, ancres)),
                  ("fin_ens", lambda: E.finales_ensemble(pop, W, ancres))]
        if choix:
            etapes.append(("fin_choix", lambda: E.choix_final(pop, W, ancres, seed)))
        for cle, f in etapes:
            if cle not in fin:
                fin = dict(fin, **{cle: f()})
                sauver(gens - 1, fin)
            out[cle] = fin[cle]
    return out


def course_seul_r(arch, seed, g_bas=0, facteur=E.FACTEUR, large=False, gens=E.GENS, finales=True, lr0=E.LR0,
                  wd=E.WD0, point=None, arret=None):
    """echelle46.course_seul, avec points de reprise."""
    dims = E.LARGE[arch] if large else {}
    with E.taille(1, **dims):
        st = _lire(point)
        if st is None:
            W = E.Monde(seed, "fixe")
            rng = np.random.default_rng(1000 + seed)
            pop = E.Seul(rng, arch, E.C_ANCRE, g_bas)
            TX, Ty = W.arrivees(E.TAMPON, 0)
            t = 0
            tr, jg, ancres = [], [], []
            g0, lr, fin = 0, None, {}
        else:
            rng, pop = st["rng"], st["pop"]
            assert pop.rng is rng
            W = _monde(seed, st["W"])
            TX, Ty, t, tr, jg, ancres, lr = st["TX"], st["Ty"], st["t"], st["tr"], st["jg"], st["ancres"], st["lr"]
            g0, fin = st["g"] + 1, st.get("fin", {})
        pj = E.juges(gens, g_bas)
        T = gens * E.PAS * facteur

        def sauver(g, fin):
            if point:
                _ecrire(point, dict(g=g, rng=rng, pop=pop, W=_etat_monde(W), TX=TX, Ty=Ty, t=t, tr=tr, jg=jg,
                                    ancres=ancres, lr=lr, fin=fin))
            if arret is not None and g == arret and not fin:
                raise Arret(g)

        for g in range(g0, gens):
            pop.regler(g)
            if g > 0:
                nx, ny = W.arrivees(E.ARRIVEES, g)
                TX = np.concatenate([TX[E.ARRIVEES:], nx]); Ty = np.concatenate([Ty[E.ARRIVEES:], ny])
            for _ in range(E.PAS * facteur):
                idx = rng.integers(0, E.TAMPON, (1, E.NB))
                lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / T))
                pop.pas_seul(TX[idx], Ty[idx], lr, wd)
                t += 1
            FX, Fy, iw = W.test_vise(g, E.C_ANCRE)        # l'ancre : la même que pour la population
            if g >= gens - E.DERNIERES:
                ancres.append((FX, Fy, iw))
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
            tr.append(dict(g=g, note=float(note[0]), lr=float(lr)))
            if g in pj:
                jg.append(E.juge_pop(pop, W, g, note, 0, ensemble=False))
            sauver(g, {})
        out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=0,
                   calcul=dict(reseaux=1, pas_par_generation=E.PAS * facteur, pas=t, dims=dims, lr0=lr0, res=E.RES,
                               **({} if wd == E.WD0 else dict(wd=wd))))
        if finales:
            if "fin_livre" not in fin:
                fl = E.mesures_finales(pop, W)
                fl["ancre"] = E.front_ancre(pop, ancres)
                fin = dict(fin, fin_livre=fl)
                sauver(gens - 1, fin)
            out["fin_livre"] = fin["fin_livre"]
    return out


def course(bras, arch, seed, gens=E.GENS, finales=True, lr0=E.LR0, res=32, wd=E.WD0, point=None, arret=None):
    """echelle46.course, pour les bras du plan de la clé 46, avec points de reprise."""
    E.resolution(res)
    gb = 10 if arch == "cnn" else 0
    if bras == "INDEPC":
        return course_pop_r(arch, seed, gb, False, gens, finales, choix=True, point=point, arret=arret)
    if bras == "SEUL":
        return course_seul_r(arch, seed, 0, E.FACTEUR, False, gens, finales, lr0, point=point, arret=arret)
    if bras == "SEUL10":
        return course_seul_r(arch, seed, 10, E.FACTEUR, False, gens, finales, lr0, point=point, arret=arret)
    if bras == "GRAND":
        return course_seul_r(arch, seed, 0, 1, True, gens, finales, lr0, point=point, arret=arret)
    if bras == "COURT":
        return course_seul_r(arch, seed, 0, 1, False, gens, finales, lr0, point=point, arret=arret)
    raise ValueError(bras)
