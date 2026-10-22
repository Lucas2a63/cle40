"""
CLÉ 46 — L'ENSEMBLE SUR L'ARCHITECTURE FORTE.

Les clés 41 à 45 n'ont porté que sur le perceptron. À seuil égal, l'ensemble des 32 réseaux de la recherche aléatoire y
bat chacune des recettes standard (clé 43), et aucune sélection essayée ne l'a établi plus haut (clés 43 à 45) ; mais à
la clé 40, sur le même monde et au seuil par défaut, un seul petit réseau convolutif finissait six à sept points
au-dessus de cet ensemble de perceptrons, et l'ensemble convolutif de la recherche aléatoire n'avait pas d'avance sur
le meilleur réseau convolutif seul. Cette clé met les deux à seuil égal, sur le réseau convolutif de la clé 40 : la
recherche aléatoire convolutive (bascule à la 10e génération), son ensemble et son choix final (INDEPC), contre le
réseau seul, le réseau seul pesé à la 10e génération, le grand réseau et le réseau court de la clé 40, à leurs taux
réglés de la clé 40 (balayage40_choix.json).

Ce fichier est le moteur de la clé 45 (echelle45.py), inchangé hors cet en-tête : les bras convolutifs (INDEPC, SEUL,
SEUL10, GRAND, COURT) y étaient déjà, comme le choix final et toutes les mesures des clés 42 à 45 (le seuil réglé sur
l'ancre, l'ensemble et ses fronts, la diversité). identite46.py vérifie qu'il refait au bit près, sur le convolutif, des
courses publiées de la clé 40.

LE CHOIX FINAL (clé 45 ; le critère : la log-vraisemblance pondérée de la bonne classe sous la probabilité moyenne de
l'ensemble choisi, sur les ancres réunies des dix dernières générations — 2 000 images et leurs poids, ceux de la note
de sélection)
  TOUS         les 32 réseaux, à poids égaux : l'ensemble de la recherche aléatoire (clés 43 et 44).
  GLOUTON      la sélection avant, avec remise (Caruana) : partir de rien, ajouter à chaque étape le réseau qui
               maximise le critère de l'ensemble ainsi formé (un réseau peut entrer plusieurs fois : son poids est
               son nombre d'entrées), pendant 64 étapes ; garder l'ensemble de l'étape où le critère est le plus haut
               (la première, à égalité).
  ELAGAGE      partir des 32 et retirer, tant que le critère monte strictement, le réseau dont le retrait le fait le
               plus monter.
  MEILLEURS8   les 8 réseaux de meilleur critère pris seuls, à poids égaux.
  MEILLEURS16  les 16 réseaux de meilleur critère pris seuls, à poids égaux.
  TETE         les d réseaux de meilleur critère pris seuls, à poids égaux, d étant le nombre de réseaux distincts de
               GLOUTON.
  HASARD       d réseaux tirés au hasard, à poids égaux (le témoin de GLOUTON à coût d'usage égal).
À égalité de critère, le plus petit indice de réseau l'emporte. L'ensemble d'une règle : la moyenne de ses
probabilités, chaque réseau compté autant de fois que sa multiplicité ; avec un biais b, la classe qui maximise
log(probabilité moyenne) + b·[classe = chat].
"""
import contextlib
import math
import os
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32

# ---------------------------------------------------------------- le monde
FATALE, P_FATALE, COUT_FATALE = 3, 0.03, 30.0
NOMS = ["avion", "auto", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]
P_MONDE = np.full(10, (1 - P_FATALE) / 9); P_MONDE[FATALE] = P_FATALE
COUT = np.ones(10); COUT[FATALE] = COUT_FATALE
POIDS_VERITE = P_MONDE * COUT / (P_MONDE * COUT).sum()
AUTRES = np.array([k for k in range(10) if k != FATALE])

# ---------------------------------------------------------------- la population
M, H, C1, C2 = 32, 128, 16, 32
NB, TAMPON, ARRIVEES, PAS = 64, 5000, 500, 100
GENS, G_DIF = 120, 30
NREMP, TEST, VISEE = 8, 200, 40
N_FLUX = 35000                                   # le reste de l'entraînement (15 000) : la réserve
BORNES = (np.log(0.01), np.log(100.0))
JUGE_EN_PLUS = {1, 2, 4, 29, 31, 33, 35, 39, 59}

# ---------------------------------------------------------------- la résolution
RES = None
D_IN = S1 = S2 = None                            # entrée du perceptron ; côtés des cartes du convolutif
PAQUET_MLP = PAQUET_CNN = None                   # images par paquet en inférence (la mémoire)
FICHIERS = None
_YY = _XX = None
_D, _SONDE, _SUITE = {}, {}, {}


def resolution(res):
    """Fixe la résolution des images (16 : celle des clés 36 à 39 ; 32 : celle de la clé 40)."""
    global RES, D_IN, S1, S2, PAQUET_MLP, PAQUET_CNN, FICHIERS, _YY, _XX
    if res == RES:
        return
    assert res in (16, 32)
    RES, D_IN, S1, S2 = res, res * res * 3, res // 2, res // 4
    PAQUET_MLP, PAQUET_CNN = (500, 200) if res == 16 else (500, 50)
    FICHIERS = tuple(f"cifar{res}_{k}.npz" for k in ("a", "b", "t"))
    _YY, _XX = np.mgrid[0:res, 0:res].astype(F32)
    _D.clear(); _SONDE.clear(); _SUITE.clear()


resolution(32)


def points_de_juge(gens):
    return sorted({g for g in range(gens) if g % 8 == 0} | {g for g in JUGE_EN_PLUS if g < gens} | {gens - 1})


# ================================================================ les données
def donnees():
    """(Xtr uint8 (50000,RES,RES,3), ytr, Xte uint8 (10000,RES,RES,3), yte), moyenne et écart-type par canal."""
    if not _D:
        d = os.path.join(ICI, "donnees")
        a, b, t = (np.load(os.path.join(d, f)) for f in FICHIERS)
        Xtr = np.concatenate([a["x"], b["x"]]); ytr = np.concatenate([a["y"], b["y"]]).astype(np.int64)
        _D["brut"] = (Xtr, ytr, t["x"], t["y"].astype(np.int64))
        if RES == 16:                                    # le calcul des clés 36 à 39, au bit près
            x = Xtr.astype(np.float64) / 255.0
            _D["moy"] = x.mean((0, 1, 2)).astype(F32)
            _D["ect"] = x.std((0, 1, 2)).astype(F32)
            del x
        else:                                            # en 32×32 : sommes entières exactes, sans copie flottante
            s1 = np.zeros(3, np.int64); s2 = np.zeros(3, np.int64)
            for i in range(0, len(Xtr), 2500):
                xi = Xtr[i:i + 2500].astype(np.int64)
                s1 += xi.sum((0, 1, 2)); s2 += (xi * xi).sum((0, 1, 2))
            n = float(Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2])
            moy = s1 / n / 255.0
            _D["moy"] = moy.astype(F32)
            _D["ect"] = np.sqrt(s2 / n / 255.0 ** 2 - moy * moy).astype(F32)
    return _D["brut"]


def normaliser(x):
    """x float32 dans [0,1], (n,RES,RES,3) -> centré-réduit par canal."""
    return (x - _D["moy"]) / _D["ect"]


# ================================================================ les transformations (banc de stress)
def tourner(x, angle):
    """Rotation bilinéaire de tout un lot (n,RES,RES,3) ; ce qui sort du cadre prend la couleur moyenne."""
    if angle == 0:
        return x
    c0 = (RES - 1) / 2
    t = np.deg2rad(angle); c, s = np.cos(t), np.sin(t)
    xi = c * (_XX - c0) + s * (_YY - c0) + c0
    yi = -s * (_XX - c0) + c * (_YY - c0) + c0
    x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
    fx, fy = (xi - x0).astype(F32), (yi - y0).astype(F32)
    out = np.zeros_like(x)
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        ok = (yy >= 0) & (yy < RES) & (xx >= 0) & (xx < RES)
        v = x[:, np.clip(yy, 0, RES - 1), np.clip(xx, 0, RES - 1), :]
        v = np.where(ok[None, :, :, None], v, _D["moy"])
        out += w[None, :, :, None] * v
    return out


def flouter(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)).astype(F32); k /= k.sum()
    for ax in (1, 2):
        p = [(0, 0)] * 4; p[ax] = (r, r)
        xp = np.pad(x, p, mode="reflect")
        out = np.zeros_like(x)
        for i, w in enumerate(k):
            out += w * (xp[:, i:i + RES] if ax == 1 else xp[:, :, i:i + RES])
        x = out
    return x


def pixelliser(x, k):
    if k >= RES:
        return x
    bas = ((np.arange(k) + 0.5) * RES / k).astype(int)
    haut = ((np.arange(RES) + 0.5) * k / RES).astype(int)
    return x[:, bas][:, :, bas][:, haut][:, :, haut]


def corrompre(x, types, s, rng):
    """Chaque image reçoit sa corruption (0 bruit, 1 flou, 2 contraste, 3 pixellisation) à la sévérité s ∈ [0,1].
    Le flou et la pixellisation sont à l'échelle de l'image : en 32×32, un flou deux fois plus large en pixels
    et une pixellisation de 32 à 16 côtés, ce qui correspond au 16×16 des clés 36 à 39."""
    if s <= 0:
        return x
    x = x.copy()
    for t in range(4):
        i = np.flatnonzero(types == t)
        if len(i) == 0:
            continue
        v = x[i]
        if t == 0:
            v = np.clip(v + rng.normal(0, 0.10 * s, v.shape).astype(F32), 0, 1)
        elif t == 1:
            v = flouter(v, 1.0 * s * RES / 16)
        elif t == 2:
            m = v.mean((1, 2, 3), keepdims=True)
            v = (v - m) * F32(1 - 0.8 * s) + m
        else:
            v = pixelliser(v, int(round(RES - RES / 2 * s)))
        x[i] = v
    return x


# ================================================================ le monde (fixe)
class Monde:
    """Les données, les tirages, et le juge. Le découpage flux / réserve est celui des clés 36 à 39."""

    def __init__(self, seed, monde="fixe"):
        assert monde == "fixe"
        Xtr, ytr, Xte, yte = donnees()
        r = np.random.default_rng(12345)                 # découpage fixe, identique partout
        perm = r.permutation(len(Xtr))
        self.flux = (Xtr[perm[:N_FLUX]], ytr[perm[:N_FLUX]])
        self.reserve = (Xtr[perm[N_FLUX:]], ytr[perm[N_FLUX:]])
        self.juge = (Xte, yte)
        self.rng = np.random.default_rng(seed)
        self.monde = monde
        self.par_classe_flux = [np.flatnonzero(self.flux[1] == c) for c in range(10)]
        self.par_classe_res = [np.flatnonzero(self.reserve[1] == c) for c in range(10)]
        self.deja = np.zeros(len(self.reserve[1]), bool)
        self.recycle = np.zeros(10, int)

    def etat(self, g):
        return 0.0

    def vue(self, X, g, rng):
        return normaliser(X.astype(F32) / F32(255))

    def arrivees(self, n, g):
        cl = self.rng.choice(10, size=n, p=P_MONDE)
        idx = np.array([self.rng.choice(self.par_classe_flux[c]) for c in cl])
        X, y = self.flux
        return self.vue(X[idx], g, self.rng), y[idx]

    def _neufs(self, classes, g):
        idx = []
        for c in classes:
            pool = self.par_classe_res[c]
            libres = pool[~self.deja[pool]]
            if len(libres) == 0:                         # classe épuisée dans la réserve : on la recycle
                self.deja[pool] = False
                self.recycle[c] += 1
                libres = pool
            i = self.rng.choice(libres)
            self.deja[i] = True
            idx.append(i)
        idx = np.array(idx)
        X, y = self.reserve
        return self.vue(X[idx], g, self.rng), y[idx]

    def test_vise(self, g, c):
        p_autres = np.where(np.arange(10) == FATALE, 0, P_MONDE) / (1 - P_FATALE)
        cl = np.concatenate([np.full(VISEE, FATALE), self.rng.choice(10, size=TEST - VISEE, p=p_autres)])
        X, y = self._neufs(cl, g)
        q = np.where(np.arange(10) == FATALE, VISEE / TEST, (1 - VISEE / TEST) * P_MONDE / (1 - P_FATALE))
        cout = np.ones(10); cout[FATALE] = c
        return X, y, (P_MONDE / q * cout)[y]            # suréchantillonné, puis repondéré : sans biais

    def juger(self, g):
        """Les 10 000 images de test (monde fixe : toujours les mêmes)."""
        X, y = self.juge
        if "juge_propre" not in _D:
            _D["juge_propre"] = self.vue(X, 0, None)
        return _D["juge_propre"], y


# ================================================================ les réseaux (toute la population à la fois)
def _patches(a, pas=1):
    """a (..., n, h, w, c) -> motifs 3×3 avec bord nul, de pas 1 ou 2 : (..., n*(h/pas)*(w/pas), 9c),
    dans l'ordre (ky, kx, c)."""
    *lead, n, h, w, c = a.shape
    p = np.zeros(tuple(lead) + (n, h + 2, w + 2, c), F32)
    p[..., 1:h + 1, 1:w + 1, :] = a
    s = p.strides
    ho, wo = h // pas, w // pas
    v = np.lib.stride_tricks.as_strided(p, shape=tuple(lead) + (n, ho, wo, 3, 3, c),
                                        strides=s[:-3] + (pas * s[-3], pas * s[-2], s[-3], s[-2], s[-1]))
    return np.ascontiguousarray(v).reshape(tuple(lead) + (n * ho * wo, 9 * c))


def _pool(a):
    """Moyenne 2×2 : (M, n, h, w, c) -> (M, n, h/2, w/2, c)."""
    return (a[:, :, 0::2, 0::2] + a[:, :, 1::2, 0::2] + a[:, :, 0::2, 1::2] + a[:, :, 1::2, 1::2]) * F32(0.25)


def _unpool(g):
    """Adjoint de _pool."""
    Mm, n, h, w, c = g.shape
    out = np.empty((Mm, n, 2 * h, 2 * w, c), F32)
    q = g * F32(0.25)
    out[:, :, 0::2, 0::2] = q; out[:, :, 1::2, 0::2] = q; out[:, :, 0::2, 1::2] = q; out[:, :, 1::2, 1::2] = q
    return out


def _retourner(K, cin, cout):
    """Noyau 3×3 retourné et transposé : ce qui ramène le gradient d'une convolution vers son entrée."""
    Mm = K.shape[0]
    return np.ascontiguousarray(K.reshape(Mm, 3, 3, cin, cout)[:, ::-1, ::-1].transpose(0, 1, 2, 4, 3)).reshape(Mm, 9 * cout, cin)


class MLP:
    """Perceptron D_IN → H → 10 (tanh)."""
    POIDS = ("W1", "b1", "W2", "b2")
    DECROIT = ("W1", "W2")

    @property
    def PAQUET(self):
        return PAQUET_MLP

    def __init__(self, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(D_IN), (M, D_IN, H)).astype(F32); self.b1 = np.zeros((M, 1, H), F32)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (M, H, 10)).astype(F32); self.b2 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        """X (n,RES,RES,3) commun à tous, ou (M,n,RES,RES,3) propre à chacun -> logits (M,n,10), traits (M,n,H)."""
        x = X.reshape(X.shape[:-3] + (D_IN,))
        h = np.tanh(np.matmul(x, self.W1) + self.b1)
        return np.matmul(h, self.W2) + self.b2, h, (x, h)

    @staticmethod
    def traits_sonde(f):
        return f

    def arriere(self, gz, cache):
        x, h = cache
        gr = {"W2": np.matmul(h.transpose(0, 2, 1), gz), "b2": gz.sum(1, keepdims=True)}
        gh = np.matmul(gz, self.W2.transpose(0, 2, 1)) * (1 - h * h)
        gr["W1"] = np.matmul(x.transpose(0, 2, 1), gh); gr["b1"] = gh.sum(1, keepdims=True)
        return gr


class CNN:
    """conv 3×3 de pas 2 (RES×RES×3 -> S1×S1×C1) -> conv 3×3 (S1×S1×C2) -> moyenne 2×2 (S2×S2×C2) -> 10, tanh."""
    POIDS = ("K1", "c1", "K2", "c2", "W3", "b3")
    DECROIT = ("K1", "K2", "W3")

    @property
    def PAQUET(self):
        return PAQUET_CNN

    def __init__(self, rng):
        self.K1 = rng.normal(0, 1 / np.sqrt(27), (M, 27, C1)).astype(F32); self.c1 = np.zeros((M, 1, C1), F32)
        self.K2 = rng.normal(0, 1 / np.sqrt(9 * C1), (M, 9 * C1, C2)).astype(F32); self.c2 = np.zeros((M, 1, C2), F32)
        self.W3 = rng.normal(0, 1 / np.sqrt(S2 * S2 * C2), (M, S2 * S2 * C2, 10)).astype(F32)
        self.b3 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        commun = X.ndim == 4
        n = X.shape[-4]
        P1 = _patches(X, pas=2)                            # (n*S1*S1, 27) ou (M, n*S1*S1, 27)
        h1 = np.tanh(np.matmul(P1, self.K1) + self.c1)    # (M, n*S1*S1, C1)
        P2 = _patches(h1.reshape(M, n, S1, S1, C1))        # (M, n*S1*S1, 9*C1)
        h2 = np.tanh(np.matmul(P2, self.K2) + self.c2)    # (M, n*S1*S1, C2)
        f = _pool(h2.reshape(M, n, S1, S1, C2)).reshape(M, n, S2 * S2 * C2)
        z = np.matmul(f, self.W3) + self.b3
        return z, f, (P1, commun, h1, P2, h2, f, n)

    @staticmethod
    def traits_sonde(f):
        """Les traits que lisent la sonde et la dimension effective : ramenés à 4×4×C2 comme aux clés 36 à 39
        (en 32×32, une moyenne 2×2 de plus ; en 16×16, rien ne change)."""
        if S2 == 4:
            return f
        Mm, n = f.shape[:2]
        return _pool(f.reshape(Mm, n, S2, S2, -1)).reshape(Mm, n, -1)

    def arriere(self, gz, cache):
        P1, commun, h1, P2, h2, f, n = cache
        gr = {"W3": np.matmul(f.transpose(0, 2, 1), gz), "b3": gz.sum(1, keepdims=True)}
        gf = np.matmul(gz, self.W3.transpose(0, 2, 1)).reshape(M, n, S2, S2, C2)
        ga2 = _unpool(gf).reshape(M, n * S1 * S1, C2) * (1 - h2 * h2)
        gr["K2"] = np.matmul(P2.transpose(0, 2, 1), ga2); gr["c2"] = ga2.sum(1, keepdims=True)
        ga1 = np.matmul(_patches(ga2.reshape(M, n, S1, S1, C2)), _retourner(self.K2, C1, C2)) * (1 - h1 * h1)
        P1t = P1.T[None] if commun else P1.transpose(0, 2, 1)
        gr["K1"] = np.matmul(P1t, ga1); gr["c1"] = ga1.sum(1, keepdims=True)
        return gr


ARCHS = {"mlp": MLP, "cnn": CNN}


# ================================================================ la population (clés 36 et 37)
class Population:
    """La population de la clé 36 ; valeur "IMP" (×c dès le départ) ou "BAS" (×c à partir de g_bas, clé 37)."""

    def __init__(self, rng, arch, valeur="IMP", c=1.0, g_bas=None, genome=None):
        self.g_bas = g_bas
        self.rng = rng
        self.net = ARCHS[arch](rng)
        # le génome (genome=None : celui des clés 36 à 41)
        gn = genome or GENOME_STD
        self.lr_borne, self.wd_borne = gn["lr_borne"], gn["wd_borne"]
        self.llr = rng.uniform(np.log(gn["lr_init"][0]), np.log(gn["lr_init"][1]), M)
        self.lwd = rng.uniform(np.log(gn["wd_init"][0]), np.log(gn["wd_init"][1]), M)
        self.eps = rng.uniform(0, 1, M)
        self.lcw = rng.normal(0, 0.1, (M, 10))
        self.u = np.zeros((M, TAMPON))                  # la curation : ce que chacun préfère revoir
        self.valeur, self.c, self.g = valeur, c, 0
        self.sans_curation = False
        self._fixer()

    def regler(self, g):
        self.g = g
        self._fixer()

    def _fixer(self):
        if self.valeur == "BAS":
            self.lcw[:] = 0.0
            self.lcw[:, FATALE] = float(self.g >= self.g_bas) * np.log(self.c)
            return
        assert self.valeur == "IMP"
        self.lcw[:] = 0.0
        self.lcw[:, FATALE] = 1.0 * np.log(self.c)

    def avant(self, X, traits=True):
        """Logits (et traits), par paquets pour ménager la mémoire."""
        taille_ = self.net.PAQUET
        if X.ndim == 4 and len(X) > taille_:
            if not traits:                               # les logits seuls : rien d'autre n'est gardé
                return np.concatenate([self.net.avant(X[i:i + taille_])[0]
                                       for i in range(0, len(X), taille_)], 1), None
            parts = [self.net.avant(X[i:i + taille_])[:2] for i in range(0, len(X), taille_)]
            return np.concatenate([p[0] for p in parts], 1), np.concatenate([p[1] for p in parts], 1)
        return self.net.avant(X)[:2]

    def predire(self, X):
        return self.avant(X, traits=False)[0].argmax(2)

    def pas(self, Xb, yb, idx):
        """Un pas de gradient pour tous les réseaux à la fois, puis la curation."""
        z, _, cache = self.net.avant(Xb)
        juste = (z.argmax(2) == yb).astype(float)
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        w = (np.take_along_axis(cw, yb, 1)[:, :, None] / NB).astype(F32)    # poids de classe dans la perte
        np.put_along_axis(p, yb[:, :, None], np.take_along_axis(p, yb[:, :, None], 2) - 1, 2)
        gr = self.net.arriere(p * w, cache)
        lr0 = np.exp(self.llr).astype(F32); wd0 = np.exp(self.lwd).astype(F32)
        for k in self.net.POIDS:
            W = getattr(self.net, k)
            forme = (M,) + (1,) * (W.ndim - 1)
            lr = lr0.reshape(forme); wd = wd0.reshape(forme)
            W -= lr * (gr[k] + wd * W) if k in self.net.DECROIT else lr * gr[k]
        # L'ABRI : ce qui est réussi devient plus probable, ce qui est échoué s'efface
        du = self.eps[:, None] * (juste - juste.mean(1, keepdims=True))
        np.add.at(self.u, (np.arange(M)[:, None].repeat(NB, 1), idx), du)
        np.clip(self.u, -8, 8, out=self.u)

    def tirer(self, n):
        """Chaque réseau tire dans le tampon selon SA curation."""
        pr = np.exp(self.u - self.u.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        cs = np.cumsum(pr, 1)
        r = self.rng.random((M, n))
        return np.stack([np.minimum(np.searchsorted(cs[m], r[m]), TAMPON - 1) for m in range(M)])

    def copier(self, dst, src, quoi="tout"):
        """La sélection : "tout" (clés 36 à 40), "poids" (l'état appris seul) ou "reglages" (le génome seul).
        Les mutations sont tirées dans tous les cas, pour que les bras partagent leurs tirages."""
        if quoi in ("tout", "poids"):
            for a in self.net.POIDS:
                getattr(self.net, a)[dst] = getattr(self.net, a)[src]
            self.u[dst] = self.u[src]
        n = len(dst); r = self.rng
        d_llr = r.normal(0, 0.2, n)                                          # L'ERREUR
        d_lwd = r.normal(0, 0.3, n)
        d_eps = r.normal(0, 0.1, n)
        d_lcw = r.normal(0, 0.15, (n, 10))
        if quoi in ("tout", "reglages"):
            self.llr[dst] = self.llr[src] + d_llr
            self.lwd[dst] = self.lwd[src] + d_lwd
            self.eps[dst] = np.clip(self.eps[src] + d_eps, 0, 3)
            self.lcw[dst] = np.clip(self.lcw[src] + d_lcw, *BORNES)
            np.clip(self.llr, np.log(self.lr_borne[0]), np.log(self.lr_borne[1]), out=self.llr)
            np.clip(self.lwd, np.log(self.wd_borne[0]), np.log(self.wd_borne[1]), out=self.lwd)
        if self.sans_curation:
            self.eps[:] = 0.0
        self._fixer()

    def poids_fatale(self):
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        return cw[:, FATALE] / cw[:, AUTRES].mean(1)


# ================================================================ les mesures (clé 36)
def rappels(pred, y):
    return np.stack([(pred[:, y == k] == k).mean(1) for k in range(10)], 1)     # (M, 10)


def lecture(pred, y):
    """Ce que le juge lit d'une population : rappels, vérité ×30, précision simple, fausses alertes."""
    rap = rappels(pred, y)
    fa = np.stack([(pred[:, y == k] == FATALE).mean(1) for k in AUTRES], 1).mean(1)
    return rap, rap @ POIDS_VERITE, rap @ P_MONDE, rap[:, AUTRES].mean(1), fa


def sonde_donnees(monde):
    """500 images propres par classe ordinaire, tirées une fois pour toutes dans la réserve."""
    if "x" not in _SONDE:
        r = np.random.default_rng(777)
        idx = np.concatenate([r.choice(monde.par_classe_res[k], 500, replace=False) for k in AUTRES])
        X, y = monde.reserve
        _SONDE["x"] = normaliser(X[idx].astype(F32) / F32(255)); _SONDE["y"] = y[idx]
    return _SONDE["x"], _SONDE["y"]


def suite_robustesse(monde):
    """Le banc de stress final : 2 000 images de test (200 par classe), propres, sous chacune des quatre
    corruptions à trois sévérités, et tournées de 15°, 30° et 45°. Tirages fixés."""
    if not _SUITE:
        X, y = monde.juge
        r = np.random.default_rng(31415)
        idx = np.concatenate([r.choice(np.flatnonzero(y == k), 200, replace=False) for k in range(10)])
        x0 = X[idx].astype(F32) / F32(255)
        _SUITE["y"] = y[idx]
        _SUITE["propre"] = normaliser(x0)
        noms = ("bruit", "flou", "contraste", "pixels")
        for t in range(4):
            for s in (1 / 3, 2 / 3, 1.0):
                _SUITE[f"{noms[t]}_{s:.2f}"] = normaliser(
                    corrompre(x0, np.full(len(x0), t), s, np.random.default_rng(900 + t)))
        for a in (15, 30, 45):
            _SUITE[f"rot_{a}"] = normaliser(tourner(x0, a))
    return _SUITE


def mesures_finales(pop, monde):
    Xj, yj = monde.juge
    JX = _D.get("juge_propre")
    if JX is None:
        JX = _D["juge_propre"] = normaliser(Xj.astype(F32) / F32(255))
    Xs, ys = sonde_donnees(monde)
    Ys = (ys[:, None] == AUTRES[None, :]).astype(np.float64)
    # la sonde : une tête linéaire neuve (moindres carrés régularisés) sur les traits gelés
    P = pop.net.PAQUET
    AtA, AtY = None, None
    for i in range(0, len(Xs), P):
        _, f = pop.avant(Xs[i:i + P])
        f = pop.net.traits_sonde(f)
        A = np.concatenate([f, np.ones(f.shape[:2] + (1,), F32)], 2).astype(np.float64)
        a, b = np.matmul(A.transpose(0, 2, 1), A), np.matmul(A.transpose(0, 2, 1), Ys[i:i + P])
        AtA = a if AtA is None else AtA + a
        AtY = b if AtY is None else AtY + b
    d = AtA.shape[1]
    Wp = np.stack([np.linalg.solve(AtA[m] + np.eye(d), AtY[m]) for m in range(M)])
    # le juge propre, par paquets : logits, sonde, et covariance des traits (dimension effective)
    Z, Ps = [], []
    S1_, S2_, n = 0.0, 0.0, 0
    for i in range(0, len(JX), P):
        z, f = pop.avant(JX[i:i + P])
        Z.append(z)
        f64 = pop.net.traits_sonde(f).astype(np.float64)
        A = np.concatenate([f64, np.ones(f.shape[:2] + (1,))], 2)
        Ps.append(AUTRES[np.matmul(A, Wp).argmax(2)])
        S1_ = S1_ + f64.sum(1); S2_ = S2_ + np.matmul(f64.transpose(0, 2, 1), f64); n += f.shape[1]
    Z = np.concatenate(Z, 1); Ps = np.concatenate(Ps, 1)
    mu = S1_ / n
    C = S2_ / n - mu[:, :, None] * mu[:, None, :]
    PR = [float(np.trace(C[m]) ** 2 / (C[m] * C[m]).sum()) for m in range(M)]
    ordi = yj != FATALE
    sonde = [float(np.mean([(Ps[m][ordi & (yj == k)] == k).mean() for k in AUTRES])) for m in range(M)]
    # le front de réglage : le seul seuil de la classe fatale déplacé
    biais = np.concatenate([[-30.0], np.linspace(-6, 14, 81)])
    Rf, Ao = [], []
    for b in biais:
        zz = Z.copy(); zz[:, :, FATALE] += b
        rap = rappels(zz.argmax(2), yj)
        Rf.append(float(rap[:, FATALE].mean())); Ao.append(float(rap[:, AUTRES].mean(1).mean()))
    rap0 = rappels(Z.argmax(2), yj).mean(0)
    # le banc de stress
    rob = {}
    S = suite_robustesse(monde)
    for k, X in S.items():
        if k == "y":
            continue
        rap, v30, simple, aord, fa = lecture(pop.predire(X), S["y"])
        rob[k] = dict(V30=float(v30.mean()), Aord=float(aord.mean()), Rf=float(rap[:, FATALE].mean()),
                      simple=float(simple.mean()))
    return dict(biais=[float(b) for b in biais], Rf=Rf, Ao=Ao, rap=[float(v) for v in rap0],
                PR=float(np.mean(PR)), PR_r=PR, sonde=float(np.mean(sonde)), sonde_r=sonde, rob=rob)


# ================================================================ les six bras (clé 39)
LR0 = math.sqrt(0.02 * 0.2)          # le centre géométrique du taux d'apprentissage de départ
WD0 = math.sqrt(1e-5 * 1e-3)         # et de la décroissance des poids
C_ANCRE = 30.0                       # le poids du chat, pour l'ancre et dans la perte
DERNIERES = 10                       # les ancres des dix dernières générations : le front de l'ancre
BIAIS = np.concatenate([[-30.0], np.linspace(-6, 14, 81)])      # la grille du front (celle de mesures_finales)
# le génome de la population : intervalles de départ (log-uniformes) et bornes du taux et de la décroissance
GENOME_STD = dict(lr_init=(0.02, 0.2), wd_init=(1e-5, 1e-3), lr_borne=(1e-3, 1.0), wd_borne=(1e-7, 1e-2))
GENOME_LARGE = dict(lr_init=(0.002, 0.2), wd_init=(1e-5, 1e-1), lr_borne=(1e-4, 1.0), wd_borne=(1e-7, 1.0))
LR_RECETTE, WD_RECETTE = LR0 / 8, 3e-2   # la recette réglée du réseau court (clé 41, balayage41_choix.json)
GENOME_RECETTE = dict(lr_init=(LR_RECETTE / math.sqrt(10), LR_RECETTE * math.sqrt(10)),
                      wd_init=(WD_RECETTE / math.sqrt(10), WD_RECETTE * math.sqrt(10)),
                      lr_borne=(1e-4, 1.0), wd_borne=(1e-7, 1.0))
FACTEUR = 32                         # le calcul de la population, en réseaux
LARGE = {"cnn": dict(C1=96, C2=186), "mlp": dict(H=4096)}
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
BRAS = ("SYS", "INDEP", "SEUL", "SEUL10", "GRAND", "COURT")


@contextlib.contextmanager
def taille(m=None, **dims):
    """Change, le temps d'un calcul, la taille de la population (M) et des réseaux (C1, C2, H)."""
    g = globals()
    avant = {k: g[k] for k in ("M", "C1", "C2", "H")}
    try:
        if m is not None:
            g["M"] = m
        for k, v in dims.items():
            g[k] = v
        yield
    finally:
        g.update(avant)


def juges(gens, g_bas):
    pj = set(points_de_juge(gens))
    if g_bas:
        pj |= {g for g in (g_bas - 1, g_bas, g_bas + 1, g_bas + 3, g_bas + 5, g_bas + 9) if 0 <= g < gens}
    return pj


def _probas(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def _lire(pred, y, best=None):
    rap, v30, simple, aord, fa = lecture(pred, y)
    d = dict(V30=float(v30.mean()), simple=float(simple.mean()), Rf=float(rap[:, FATALE].mean()),
             Aord=float(aord.mean()), FA=float(fa.mean()), gel=float((fa > 0.5).mean()))
    if best is not None:
        d.update(V30_b=float(v30[best]), simple_b=float(simple[best]), Rf_b=float(rap[best, FATALE]),
                 Aord_b=float(aord[best]), FA_b=float(fa[best]))
    return d


def lire_ensemble(z, y):
    """L'ensemble : la moyenne des probabilités des réseaux, puis la classe la plus probable."""
    pe = _probas(z).mean(0)
    d = _lire(pe.argmax(1)[None], y)
    return {k + "_ens": v for k, v in d.items() if k != "gel"}


def probas_ensemble(pop, X):
    """La probabilité moyenne des réseaux de la population (l'ensemble), en double précision. Ne consomme aucun
    tirage."""
    return _probas(pop.avant(X, traits=False)[0]).mean(0)


def decide_ensemble(pe, b):
    """La décision de l'ensemble avec un biais b sur le chat : argmax de log(probabilité moyenne) + b·[chat]."""
    lp = np.log(np.maximum(pe, 1e-300))
    lp[:, FATALE] += b
    return lp.argmax(1)


def note_ensemble(pop, FX, Fy, iw):
    """La note de l'ancre de l'ensemble (pondérée, sans biais), sur l'ancre d'une génération."""
    return float(((decide_ensemble(probas_ensemble(pop, FX), 0.0) == Fy) * iw).sum() / iw.sum())


def diversite(z, y):
    """La diversité d'une population sur des images (clé 44), à partir de ses logits z (réseaux × images × classes) :
    l'ambiguïté — l'écart quadratique moyen des probabilités des réseaux à leur moyenne, sur toutes les images et
    sur les seuls chats —, le désaccord — la part des (réseau, image) dont la classe la plus probable diffère de celle
    de l'ensemble —, et les scores de Brier moyen des réseaux et de l'ensemble (Brier de l'ensemble = Brier moyen −
    ambiguïté). Ne consomme aucun tirage."""
    P = _probas(z)
    pe = P.mean(0)
    Y = np.zeros_like(pe)
    Y[np.arange(len(y)), y] = 1.0
    ec = ((P - pe[None]) ** 2).sum(-1)
    chat = y == FATALE
    return dict(amb=float(ec.mean()), amb_chat=float(ec[:, chat].mean()),
                desacc=float((P.argmax(-1) != pe.argmax(-1)[None]).mean()),
                brier_moy=float(((P - Y[None]) ** 2).sum(-1).mean()), brier_ens=float(((pe - Y) ** 2).sum(-1).mean()))


def contributions(pop, FX, Fy, iw):
    """Ce que chaque réseau apporte à l'ensemble sur l'ancre d'une génération (clé 44) : la log-vraisemblance pondérée
    (les poids de l'ancre) de la bonne classe sous la probabilité moyenne des 32 réseaux, moins la même sans lui.
    Rend les apports et la log-vraisemblance de l'ensemble. Ne consomme aucun tirage."""
    P = _probas(pop.avant(FX, traits=False)[0])
    py = P[:, np.arange(len(Fy)), Fy]
    s = py.sum(0)
    w = iw / iw.sum()
    tout = float((w * np.log(np.maximum(s / M, 1e-300))).sum())
    sans = (w[None] * np.log(np.maximum((s[None] - py) / (M - 1), 1e-300))).sum(1)
    return tout - sans, tout


def juge_pop(pop, W, g, note, best, ensemble):
    """Le juge à la génération g : la population (moyenne), le réseau livré, et l'ensemble."""
    JX, Jy = W.juger(g)
    z = pop.avant(JX, traits=False)[0]
    d = dict(g=g, etat=float(W.etat(g)), note_b=float(note[best]))
    d.update(_lire(z.argmax(2), Jy, best))
    if ensemble:
        d.update(lire_ensemble(z, Jy))
        d.update(diversite(z, Jy))                        # clé 44 : la diversité de la population
    return d


def vue(pop, k):
    """Une population d'un seul réseau : le réseau k, ses poids copiés (pour mesures_finales)."""
    sub = object.__new__(Population)
    net = object.__new__(type(pop.net))
    for a in pop.net.POIDS:
        setattr(net, a, getattr(pop.net, a)[k:k + 1].copy())
    sub.net = net
    return sub


def front_ancre(pop1, ancres):
    """Le front de l'ancre d'un seul réseau (pop1, population d'un réseau) : la note de l'ancre — pondérée, sans
    biais, comme pour la sélection — sur les ancres réunies des dernières générations, pour chaque biais ajouté au
    logit du chat (la grille du front du juge). Ne consomme aucun tirage."""
    X = np.concatenate([a[0] for a in ancres]); y = np.concatenate([a[1] for a in ancres])
    w = np.concatenate([a[2] for a in ancres])
    z = pop1.avant(X, traits=False)[0][0].astype(np.float64)
    notes = []
    for b in BIAIS:
        zz = z.copy(); zz[:, FATALE] += b
        notes.append(float(((zz.argmax(1) == y) * w).sum() / w.sum()))
    return dict(biais=[float(b) for b in BIAIS], note=notes, n=int(len(y)), chats=int((y == FATALE).sum()))


def finales_reseau(pop, W, k, ancres=None):
    with taille(1):
        out = mesures_finales(vue(pop, k), W)
        if ancres:
            out["ancre"] = front_ancre(vue(pop, k), ancres)
        return out


def finales_ensemble(pop, W, ancres=None):
    """Le juge propre et le banc de stress de la clé 36, lus par l'ensemble des réseaux ; puis (clé 43) le front du
    juge de l'ensemble et, si les ancres sont données, son front de l'ancre. Ne consomme aucun tirage."""
    JX = _D.get("juge_propre")
    if JX is None:
        Xj, _ = W.juge
        JX = _D["juge_propre"] = normaliser(Xj.astype(F32) / F32(255))
    zj = pop.avant(JX, traits=False)[0]
    out = lire_ensemble(zj, W.juge[1])
    rob = {}
    S = suite_robustesse(W)
    for k, X in S.items():
        if k == "y":
            continue
        d = lire_ensemble(pop.avant(X, traits=False)[0], S["y"])
        rob[k] = dict(V30=d["V30_ens"], Aord=d["Aord_ens"], Rf=d["Rf_ens"], simple=d["simple_ens"])
    out["rob"] = rob
    # clé 43 : le front du juge de l'ensemble (le seul seuil du chat déplacé)
    pe, yj = _probas(zj).mean(0), W.juge[1]
    Rf, Ao = [], []
    for b in BIAIS:
        rap = rappels(decide_ensemble(pe, b)[None], yj)
        Rf.append(float(rap[0, FATALE])); Ao.append(float(rap[0, AUTRES].mean()))
    out["front"] = dict(biais=[float(b) for b in BIAIS], Rf=Rf, Ao=Ao)
    out["diversite"] = diversite(zj, yj)                  # clé 44 : la diversité de la population finale
    # clé 43 : le front de l'ancre de l'ensemble, sur les ancres des dernières générations réunies
    if ancres:
        X = np.concatenate([a[0] for a in ancres]); y = np.concatenate([a[1] for a in ancres])
        w = np.concatenate([a[2] for a in ancres])
        pa = probas_ensemble(pop, X)
        notes = [float(((decide_ensemble(pa, b) == y) * w).sum() / w.sum()) for b in BIAIS]
        out["ancre"] = dict(biais=[float(b) for b in BIAIS], note=notes, n=int(len(y)), chats=int((y == FATALE).sum()))
    return out


# ================================================================ clé 45 : choisir l'ensemble à la fin
REGLES = ("TOUS", "GLOUTON", "ELAGAGE", "MEILLEURS8", "MEILLEURS16", "TETE", "HASARD")
ETAPES_GLOUTON = 64


def critere_ensemble(Q, W, m):
    """Le critère d'un ensemble de multiplicités m : Σ_i W_i log(Σ_k m_k Q[k, i] / Σ_k m_k), où Q[k, i] est la
    probabilité que le réseau k donne à la bonne classe de l'image i des ancres réunies, et W les poids de l'ancre
    normalisés (de somme 1)."""
    s = (m[:, None].astype(np.float64) * Q).sum(0) / m.sum()
    return float((W * np.log(np.maximum(s, 1e-300))).sum())


def glouton(Q, W, etapes=ETAPES_GLOUTON):
    """La sélection avant, avec remise (Caruana et coll., 2004) : renvoie les multiplicités de l'ensemble retenu, le
    critère à chaque étape et l'étape retenue (la première où le critère est le plus haut)."""
    nr, n = Q.shape
    s = np.zeros(n)
    m = np.zeros(nr, dtype=np.int64)
    chemin, etats = [], []
    for t in range(1, etapes + 1):
        cand = (W[None] * np.log(np.maximum((s[None] + Q) / t, 1e-300))).sum(1)
        k = int(np.argmax(cand))
        s = s + Q[k]
        m[k] += 1
        chemin.append(float(cand[k]))
        etats.append(m.copy())
    t0 = int(np.argmax(chemin))
    return etats[t0], chemin, t0 + 1


def elagage(Q, W):
    """L'élagage : partir de tous les réseaux et retirer, tant que le critère monte strictement, le réseau dont le
    retrait le fait le plus monter. Renvoie les multiplicités (0 ou 1) et le critère à chaque étape."""
    nr = len(Q)
    m = np.ones(nr, dtype=np.int64)
    s = Q.sum(0)
    n = nr
    cur = float((W * np.log(np.maximum(s / n, 1e-300))).sum())
    chemin = [cur]
    while n > 1:
        idx = np.flatnonzero(m)
        vals = (W[None] * np.log(np.maximum((s[None] - Q[idx]) / (n - 1), 1e-300))).sum(1)
        j = int(np.argmax(vals))
        if not vals[j] > cur:
            break
        k = int(idx[j])
        m[k] = 0
        s = s - Q[k]
        n -= 1
        cur = float(vals[j])
        chemin.append(cur)
    return m, chemin


def choix_final(pop, W, ancres, seed):
    """Clé 45 : les sept règles de choix de l'ensemble, appliquées aux réseaux finaux d'une population. Pour chaque
    règle : les multiplicités, le nombre de réseaux distincts, le critère sur les ancres réunies, le front du juge
    (le seul seuil du chat déplacé), le front de l'ancre (la note pondérée sur les ancres réunies, pour chaque biais)
    et la diversité sur le juge. Pour TOUS, le front et le front de l'ancre sont, au bit près, ceux de fin_ens.
    Ne consomme aucun tirage de la course."""
    JX = _D.get("juge_propre")
    if JX is None:
        Xj, _ = W.juge
        JX = _D["juge_propre"] = normaliser(Xj.astype(F32) / F32(255))
    yj = W.juge[1]
    zj = pop.avant(JX, traits=False)[0]
    Pj = _probas(zj)
    Xa = np.concatenate([a[0] for a in ancres]); ya = np.concatenate([a[1] for a in ancres])
    wa = np.concatenate([a[2] for a in ancres])
    Pa = _probas(pop.avant(Xa, traits=False)[0])
    nr = len(Pa)
    Q = Pa[:, np.arange(len(ya)), ya]
    Wn = wa / wa.sum()
    indiv = (Wn[None] * np.log(np.maximum(Q, 1e-300))).sum(1)
    ordre = np.argsort(-indiv, kind="stable")
    regles = {}
    regles["TOUS"] = np.ones(nr, dtype=np.int64)
    m_g, chemin_g, etape_g = glouton(Q, Wn)
    regles["GLOUTON"] = m_g
    m_e, chemin_e = elagage(Q, Wn)
    regles["ELAGAGE"] = m_e
    for k in (8, 16):
        m = np.zeros(nr, dtype=np.int64)
        m[ordre[:k]] = 1
        regles[f"MEILLEURS{k}"] = m
    d = int((m_g > 0).sum())
    m = np.zeros(nr, dtype=np.int64)
    m[ordre[:d]] = 1
    regles["TETE"] = m
    m = np.zeros(nr, dtype=np.int64)
    m[np.random.default_rng(20000 + seed).choice(nr, d, replace=False)] = 1
    regles["HASARD"] = m
    rap, v30, simple, aord, fa = lecture(Pj.argmax(2), yj)
    out = dict(indiv=[float(x) for x in indiv],
               reseaux=dict(V30=[float(x) for x in v30], Rf=[float(x) for x in rap[:, FATALE]],
                            Ao=[float(x) for x in rap[:, AUTRES].mean(1)],
                            lr=[float(x) for x in np.exp(pop.llr)], wd=[float(x) for x in np.exp(pop.lwd)],
                            eps=[float(x) for x in pop.eps]),
               glouton=dict(chemin=chemin_g, etape=etape_g), elagage=dict(chemin=chemin_e),
               ancres=dict(n=int(len(ya)), chats=int((ya == FATALE).sum())), regles={})
    for nom in REGLES:
        m = regles[nom]
        idx = np.repeat(np.arange(nr), m)
        pe, pa = Pj[idx].mean(0), Pa[idx].mean(0)
        Rf, Ao, note = [], [], []
        for b in BIAIS:
            r_ = rappels(decide_ensemble(pe, b)[None], yj)
            Rf.append(float(r_[0, FATALE]))
            Ao.append(float(r_[0, AUTRES].mean()))
            note.append(float(((decide_ensemble(pa, b) == ya) * wa).sum() / wa.sum()))
        out["regles"][nom] = dict(m=[int(x) for x in m], nb=int((m > 0).sum()), crit=critere_ensemble(Q, Wn, m),
                                  front=dict(biais=[float(b) for b in BIAIS], Rf=Rf, Ao=Ao),
                                  ancre=dict(biais=[float(b) for b in BIAIS], note=note),
                                  diversite=diversite(zj[idx], yj))
    return out


def course_pop(arch, seed, g_bas, selection=True, gens=GENS, finales=True, quoi="tout", sans_curation=False,
               genome=None, critere="note", choix=False):
    """SYS (selection=True) ; INDEP : la même, sans sélection. g_bas = 0 : ×30 dès le départ ; sinon la bascule.
    quoi : ce que copie la sélection ("tout", "poids", "reglages") ; sans_curation : force de curation nulle."""
    W = Monde(seed, "fixe")
    rng = np.random.default_rng(1000 + seed)
    if g_bas:
        pop = Population(rng, arch, "BAS", C_ANCRE, g_bas, genome=genome)
    else:
        pop = Population(rng, arch, "IMP", C_ANCRE, genome=genome)
    ancres = []
    if sans_curation:
        pop.sans_curation = True
        pop.eps[:] = 0.0
    TX, Ty = W.arrivees(TAMPON, 0)
    pj = juges(gens, g_bas)
    anc = np.arange(M)
    tr, jg = [], []
    for g in range(gens):
        pop.regler(g)
        if g > 0:                                         # le monde envoie du neuf (FIFO)
            nx, ny = W.arrivees(ARRIVEES, g)
            TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            pop.u = np.concatenate([pop.u[:, ARRIVEES:],
                                    np.repeat(pop.u.mean(1, keepdims=True), ARRIVEES, 1)], 1)
        for _ in range(PAS):
            idx = pop.tirer(NB)
            pop.pas(TX[idx], Ty[idx], idx)
        FX, Fy, iw = W.test_vise(g, C_ANCRE)              # l'ancre
        if g >= gens - DERNIERES:
            ancres.append((FX, Fy, iw))
        note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean()), llr_b=float(pop.llr[best]), lwd_b=float(pop.lwd[best]),
                       eps_b=float(pop.eps[best]), livre=best))
        bas, haut = o[:NREMP], o[-NREMP:]                 # la sélection : selon la note propre…
        if critere == "ensemble":                         # … ou (clé 44) selon l'apport à l'ensemble
            apport, lv = contributions(pop, FX, Fy, iw)
            oc = np.argsort(apport, kind="stable")
            haut = oc[-NREMP:]                            # copiés : le quart qui apporte le plus
            bas = oc[oc != best][:NREMP]                  # remplacés : le quart qui apporte le moins (hors le livré)
            tr[-1].update(lv_ens=lv, apport_min=float(apport.min()), apport_max=float(apport.max()),
                          apport_livre=float(apport[best]), livre_protege=int(best in oc[:NREMP]),
                          llr_sd=float(pop.llr.std()), lwd_sd=float(pop.lwd.std()))
        if g in pj:                                       # le juge, que le système ne voit jamais
            d = juge_pop(pop, W, g, note, best, ensemble=True)
            d["lignees"] = int(len(np.unique(anc)))
            d["note_ens"] = note_ensemble(pop, FX, Fy, iw)  # clé 43 : la note de l'ancre de l'ensemble
            jg.append(d)
        if selection:                                     # la sélection (SYS, et les bras qui en dérivent)
            src = rng.choice(haut, NREMP, replace=False)
            pop.copier(bas, src, quoi)
            if quoi in ("tout", "poids"):                 # les lignées suivent les poids
                anc[bas] = anc[src]
    out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=best,
               calcul=dict(reseaux=M, pas_par_generation=M * PAS, pas=M * PAS * gens, res=RES))
    if genome is not None:
        out["genome"] = {k: [float(x) for x in v] for k, v in genome.items()}
    if finales:
        out["fin"] = mesures_finales(pop, W)              # la population (moyenne)
        out["fin_livre"] = finales_reseau(pop, W, best, ancres)   # le réseau livré, et son front de l'ancre
        out["fin_ens"] = finales_ensemble(pop, W, ancres)  # l'ensemble des 32, et (clé 43) ses fronts
        if choix:                                         # clé 45 : choisir l'ensemble à la fin
            out["fin_choix"] = choix_final(pop, W, ancres, seed)
    return out


class Seul(Population):
    """Un réseau seul, réglages fixes : décroissance WD0, pas de curation, tirage uniforme dans le tampon,
    taux d'apprentissage donné à chaque pas (le cosinus) ; ×c dans la perte à partir de g_bas."""

    def __init__(self, rng, arch, c, g_bas):
        self.rng = rng
        self.net = ARCHS[arch](rng)
        self.llr = np.full(1, math.log(LR0)); self.lwd = np.full(1, math.log(WD0))
        self.eps = np.zeros(1); self.lcw = np.zeros((1, 10)); self.u = np.zeros((1, TAMPON))
        self.valeur, self.c, self.g, self.g_bas = "BAS", c, 0, g_bas
        self._fixer()

    def _fixer(self):
        self.lcw[:] = 0.0
        self.lcw[:, FATALE] = float(self.g >= self.g_bas) * math.log(self.c)

    def pas_seul(self, Xb, yb, lr, wd=WD0):
        """Le pas de gradient de la population, pour un réseau, au taux lr."""
        z, _, cache = self.net.avant(Xb)
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        w = (np.take_along_axis(cw, yb, 1)[:, :, None] / NB).astype(F32)
        np.put_along_axis(p, yb[:, :, None], np.take_along_axis(p, yb[:, :, None], 2) - 1, 2)
        gr = self.net.arriere(p * w, cache)
        lr = np.full((1, 1, 1), lr, F32); wd = np.full((1, 1, 1), wd, F32)
        for k in self.net.POIDS:
            Wk = getattr(self.net, k)
            Wk -= lr * (gr[k] + wd * Wk) if k in self.net.DECROIT else lr * gr[k]


def course_seul(arch, seed, g_bas=0, facteur=FACTEUR, large=False, gens=GENS, finales=True, lr0=LR0, wd=WD0):
    """SEUL (facteur 32), SEUL10 (g_bas 10), GRAND (large, facteur 1), COURT (facteur 1) ; lr0 : le taux de
    départ du cosinus (choisi pour chaque bras par le balayage)."""
    dims = LARGE[arch] if large else {}
    with taille(1, **dims):
        W = Monde(seed, "fixe")
        rng = np.random.default_rng(1000 + seed)
        pop = Seul(rng, arch, C_ANCRE, g_bas)
        TX, Ty = W.arrivees(TAMPON, 0)
        pj = juges(gens, g_bas)
        T = gens * PAS * facteur
        t = 0
        tr, jg, ancres = [], [], []
        for g in range(gens):
            pop.regler(g)
            if g > 0:
                nx, ny = W.arrivees(ARRIVEES, g)
                TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            for _ in range(PAS * facteur):
                idx = rng.integers(0, TAMPON, (1, NB))
                lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / T))
                pop.pas_seul(TX[idx], Ty[idx], lr, wd)
                t += 1
            FX, Fy, iw = W.test_vise(g, C_ANCRE)          # l'ancre : la même que pour la population
            if g >= gens - DERNIERES:
                ancres.append((FX, Fy, iw))
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
            tr.append(dict(g=g, note=float(note[0]), lr=float(lr)))
            if g in pj:
                jg.append(juge_pop(pop, W, g, note, 0, ensemble=False))
        out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=0,
                   calcul=dict(reseaux=1, pas_par_generation=PAS * facteur, pas=t, dims=dims, lr0=lr0, res=RES,
                               **({} if wd == WD0 else dict(wd=wd))))
        if finales:
            out["fin_livre"] = mesures_finales(pop, W)
            out["fin_livre"]["ancre"] = front_ancre(pop, ancres)
    return out


def course_meilleur32(arch, seed, lr0, wd, gens=GENS, finales=True):
    """M32 : 32 recettes standard en parallèle — taux en cosinus de lr0 à 0 sur toute la course, décroissance wd,
    tirage uniforme dans le tampon (curation nulle), ×30 dès le départ —, sans sélection ; on livre la mieux
    notée par l'ancre à la dernière génération. Même calcul que le système (32 × 100 pas par génération)."""
    W = Monde(seed, "fixe")
    rng = np.random.default_rng(1000 + seed)
    pop = Population(rng, arch, "IMP", C_ANCRE)
    pop.sans_curation = True
    pop.eps[:] = 0.0
    pop.lwd[:] = math.log(wd)
    TX, Ty = W.arrivees(TAMPON, 0)
    pj = juges(gens, 0)
    T = gens * PAS
    t = 0
    tr, jg, ancres = [], [], []
    for g in range(gens):
        pop.regler(g)
        if g > 0:
            nx, ny = W.arrivees(ARRIVEES, g)
            TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
        for _ in range(PAS):
            lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / T))
            pop.llr[:] = math.log(lr)
            idx = pop.tirer(NB)
            pop.pas(TX[idx], Ty[idx], idx)
            t += 1
        FX, Fy, iw = W.test_vise(g, C_ANCRE)
        if g >= gens - DERNIERES:
            ancres.append((FX, Fy, iw))
        note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), lr=float(lr), livre=best))
        if g in pj:
            d = juge_pop(pop, W, g, note, best, ensemble=True)
            d["lignees"] = M
            d["note_ens"] = note_ensemble(pop, FX, Fy, iw)  # clé 43 : la note de l'ancre de l'ensemble
            jg.append(d)
    out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=best,
               calcul=dict(reseaux=M, pas_par_generation=M * PAS, pas=M * PAS * gens, lr0=lr0, wd=wd, res=RES))
    if finales:
        out["fin"] = mesures_finales(pop, W)
        out["fin_livre"] = finales_reseau(pop, W, best, ancres)
        out["fin_ens"] = finales_ensemble(pop, W, ancres)  # l'ensemble des 32, et (clé 43) ses fronts
    return out


def course(bras, arch, seed, gens=GENS, finales=True, lr0=LR0, res=32, wd=WD0):
    """Une course du plan. lr0 et wd ne servent qu'aux bras de recette standard (SEUL…, GRAND…, COURT…, M32)."""
    resolution(res)
    gb = 10 if arch == "cnn" else 0                     # la bascule du système (contrainte 31)
    if bras == "SYS":
        return course_pop(arch, seed, gb, True, gens, finales)
    if bras == "INDEP":
        return course_pop(arch, seed, gb, False, gens, finales)
    if bras == "INDEPC":                                # clé 45 : la recherche aléatoire, puis le choix final
        return course_pop(arch, seed, gb, False, gens, finales, choix=True)
    if bras == "INDEP0":
        return course_pop(arch, seed, gb, False, gens, finales, sans_curation=True)
    if bras == "SEUL":
        return course_seul(arch, seed, 0, FACTEUR, False, gens, finales, lr0)
    if bras == "SEUL10":
        return course_seul(arch, seed, 10, FACTEUR, False, gens, finales, lr0)
    if bras == "GRAND":
        return course_seul(arch, seed, 0, 1, True, gens, finales, lr0)
    if bras == "COURT":
        return course_seul(arch, seed, 0, 1, False, gens, finales, lr0)
    if bras == "POIDS":
        return course_pop(arch, seed, gb, True, gens, finales, quoi="poids")
    if bras == "REGLAGES":
        return course_pop(arch, seed, gb, True, gens, finales, quoi="reglages")
    if bras == "SANSCUR":
        return course_pop(arch, seed, gb, True, gens, finales, sans_curation=True)
    if bras == "SYSL":
        return course_pop(arch, seed, gb, True, gens, finales, genome=GENOME_LARGE)
    if bras == "SYSR":
        return course_pop(arch, seed, gb, True, gens, finales, genome=GENOME_RECETTE)
    if bras == "M32":
        return course_meilleur32(arch, seed, lr0, wd, gens, finales)
    if bras == "ENS":
        return course_pop(arch, seed, gb, True, gens, finales, critere="ensemble")
    if bras == "ENSR":
        return course_pop(arch, seed, gb, True, gens, finales, quoi="reglages", critere="ensemble")
    if bras == "SEUL2":
        return course_seul(arch, seed, 0, FACTEUR, False, gens, finales, lr0, wd)
    if bras == "GRAND2":
        return course_seul(arch, seed, 0, 1, True, gens, finales, lr0, wd)
    if bras == "COURT2":
        return course_seul(arch, seed, 0, 1, False, gens, finales, lr0, wd)
    raise ValueError(bras)
