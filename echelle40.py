"""
CLÉ 40 — L'ÉCHELLE : LE MÊME MONDE, EN PLEINE RÉSOLUTION.

La clé 39 a mis le système entier face à la recette standard, à calcul d'entraînement égal, sur CIFAR-10
réduit en 16×16. Cette clé refait la même comparaison sur les images d'origine, en 32×32 : quatre fois plus
de pixels par image, des réseaux quatre fois plus coûteux par image, le même nombre de pas. Tout le reste
est inchangé : le chat fatal (3 % du monde, ×30 pour le juge), le flux (500 images neuves par génération
dans un tampon de 5 000), l'ancre (200 images neuves de la réserve, dont 40 chats, pesées au coût), le juge
(les 10 000 images de test), 120 générations, monde fixe, les six bras.

Ce fichier réunit, sans rien changer à leurs calculs, le moteur de la clé 36 (echelle.py), la bascule de la
clé 37 (echelle37.py) et les six bras de la clé 39 (echelle39.py), avec une seule chose de plus : la
résolution RES (16 ou 32) n'est plus écrite en dur. En 16×16, il refait au bit près les courses de la clé
39 (controle40.py le vérifie). Les images 16×16 des clés 36 à 39 sont exactement la moyenne arrondie de
chaque carré 2×2 de ces images 32×32, dans le même ordre : à graine égale, une course de la clé 40 voit
les mêmes images, dans le même ordre, que la course de la clé 39 — en quatre fois plus fin.

LES RÉSEAUX (en 32×32)
  mlp  perceptron 3 072 → 128 → 10 (tanh)
  cnn  conv 3×3 de pas 2 (32×32×3 → 16×16×16) → conv 3×3 (16×16×32) → moyenne 2×2 (8×8×32) → 2 048 → 10
  GRAND : convolutif à 96 et 186 filtres, perceptron à 4 096 neurones — 32 fois plus coûteux par image
  (31,98 fois en inférence et 32,74 fois à l'entraînement pour le convolutif, 32,00 pour le perceptron).

LES BRAS (ceux de la clé 39)
  SYS    le système entier : 32 réseaux, l'ancre visée sur le chat, la sélection (le quart le moins bien noté
         remplacé par des copies mutées du quart le mieux noté), ×30 dans la perte à la génération 10 pour le
         convolutif, dès 0 pour le perceptron ; il livre le réseau le mieux noté par l'ancre.
  INDEP  la même population sans sélection (une recherche aléatoire de réglages).
  SEUL   la recette standard : un réseau, tout le calcul (3 200 pas par génération), ×30 dès le départ, taux en
         cosinus de lr0 à 0 (lr0 réglé par un balayage sur la graine 99, noté par l'ancre seule).
  SEUL10 SEUL avec la bascule à la génération 10 (convolutif).
  GRAND  la recette standard avec un réseau 32 fois plus coûteux, 100 pas par génération.
  COURT  SEUL avec 100 pas par génération : le calcul d'un seul membre de la population.
L'unité de calcul : un pas de gradient d'un réseau sur un lot de 64 images.
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

    def __init__(self, rng, arch, valeur="IMP", c=1.0, g_bas=None):
        self.g_bas = g_bas
        self.rng = rng
        self.net = ARCHS[arch](rng)
        # le génome
        self.llr = rng.uniform(np.log(0.02), np.log(0.2), M)
        self.lwd = rng.uniform(np.log(1e-5), np.log(1e-3), M)
        self.eps = rng.uniform(0, 1, M)
        self.lcw = rng.normal(0, 0.1, (M, 10))
        self.u = np.zeros((M, TAMPON))                  # la curation : ce que chacun préfère revoir
        self.valeur, self.c, self.g = valeur, c, 0
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

    def copier(self, dst, src):
        for a in self.net.POIDS:
            getattr(self.net, a)[dst] = getattr(self.net, a)[src]
        self.u[dst] = self.u[src]
        n = len(dst); r = self.rng
        self.llr[dst] = self.llr[src] + r.normal(0, 0.2, n)                  # L'ERREUR
        self.lwd[dst] = self.lwd[src] + r.normal(0, 0.3, n)
        self.eps[dst] = np.clip(self.eps[src] + r.normal(0, 0.1, n), 0, 3)
        self.lcw[dst] = np.clip(self.lcw[src] + r.normal(0, 0.15, (n, 10)), *BORNES)
        np.clip(self.llr, np.log(1e-3), np.log(1.0), out=self.llr)
        np.clip(self.lwd, np.log(1e-7), np.log(1e-2), out=self.lwd)
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


def juge_pop(pop, W, g, note, best, ensemble):
    """Le juge à la génération g : la population (moyenne), le réseau livré, et l'ensemble."""
    JX, Jy = W.juger(g)
    z = pop.avant(JX, traits=False)[0]
    d = dict(g=g, etat=float(W.etat(g)), note_b=float(note[best]))
    d.update(_lire(z.argmax(2), Jy, best))
    if ensemble:
        d.update(lire_ensemble(z, Jy))
    return d


def vue(pop, k):
    """Une population d'un seul réseau : le réseau k, ses poids copiés (pour mesures_finales)."""
    sub = object.__new__(Population)
    net = object.__new__(type(pop.net))
    for a in pop.net.POIDS:
        setattr(net, a, getattr(pop.net, a)[k:k + 1].copy())
    sub.net = net
    return sub


def finales_reseau(pop, W, k):
    with taille(1):
        return mesures_finales(vue(pop, k), W)


def finales_ensemble(pop, W):
    """Le juge propre et le banc de stress de la clé 36, lus par l'ensemble des réseaux."""
    JX = _D.get("juge_propre")
    if JX is None:
        Xj, _ = W.juge
        JX = _D["juge_propre"] = normaliser(Xj.astype(F32) / F32(255))
    out = lire_ensemble(pop.avant(JX, traits=False)[0], W.juge[1])
    rob = {}
    S = suite_robustesse(W)
    for k, X in S.items():
        if k == "y":
            continue
        d = lire_ensemble(pop.avant(X, traits=False)[0], S["y"])
        rob[k] = dict(V30=d["V30_ens"], Aord=d["Aord_ens"], Rf=d["Rf_ens"], simple=d["simple_ens"])
    out["rob"] = rob
    return out


def course_pop(arch, seed, g_bas, selection=True, gens=GENS, finales=True):
    """SYS (selection=True) ; INDEP : la même, sans sélection. g_bas = 0 : ×30 dès le départ ; sinon la bascule."""
    W = Monde(seed, "fixe")
    rng = np.random.default_rng(1000 + seed)
    if g_bas:
        pop = Population(rng, arch, "BAS", C_ANCRE, g_bas)
    else:
        pop = Population(rng, arch, "IMP", C_ANCRE)
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
        note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean()), llr_b=float(pop.llr[best]), lwd_b=float(pop.lwd[best]),
                       eps_b=float(pop.eps[best]), livre=best))
        if g in pj:                                       # le juge, que le système ne voit jamais
            d = juge_pop(pop, W, g, note, best, ensemble=True)
            d["lignees"] = int(len(np.unique(anc)))
            jg.append(d)
        if selection:                                     # la sélection (SYS seulement)
            src = rng.choice(o[-NREMP:], NREMP, replace=False)
            pop.copier(o[:NREMP], src)
            anc[o[:NREMP]] = anc[src]
    out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=best,
               calcul=dict(reseaux=M, pas_par_generation=M * PAS, pas=M * PAS * gens, res=RES))
    if finales:
        out["fin"] = mesures_finales(pop, W)              # la population (moyenne)
        out["fin_livre"] = finales_reseau(pop, W, best)   # le réseau livré
        out["fin_ens"] = finales_ensemble(pop, W)         # l'ensemble des 32
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


def course_seul(arch, seed, g_bas=0, facteur=FACTEUR, large=False, gens=GENS, finales=True, lr0=LR0):
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
        tr, jg = [], []
        for g in range(gens):
            pop.regler(g)
            if g > 0:
                nx, ny = W.arrivees(ARRIVEES, g)
                TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            for _ in range(PAS * facteur):
                idx = rng.integers(0, TAMPON, (1, NB))
                lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / T))
                pop.pas_seul(TX[idx], Ty[idx], lr)
                t += 1
            FX, Fy, iw = W.test_vise(g, C_ANCRE)          # l'ancre : la même que pour la population
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
            tr.append(dict(g=g, note=float(note[0]), lr=float(lr)))
            if g in pj:
                jg.append(juge_pop(pop, W, g, note, 0, ensemble=False))
        out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=0,
                   calcul=dict(reseaux=1, pas_par_generation=PAS * facteur, pas=t, dims=dims, lr0=lr0, res=RES))
        if finales:
            out["fin_livre"] = mesures_finales(pop, W)
    return out


def course(bras, arch, seed, gens=GENS, finales=True, lr0=LR0, res=32):
    """Une course du plan. lr0 ne sert qu'aux bras d'un réseau seul."""
    resolution(res)
    gb = 10 if arch == "cnn" else 0                     # la bascule du système (contrainte 31)
    if bras == "SYS":
        return course_pop(arch, seed, gb, True, gens, finales)
    if bras == "INDEP":
        return course_pop(arch, seed, gb, False, gens, finales)
    if bras == "SEUL":
        return course_seul(arch, seed, 0, FACTEUR, False, gens, finales, lr0)
    if bras == "SEUL10":
        return course_seul(arch, seed, 10, FACTEUR, False, gens, finales, lr0)
    if bras == "GRAND":
        return course_seul(arch, seed, 0, 1, True, gens, finales, lr0)
    if bras == "COURT":
        return course_seul(arch, seed, 0, 1, False, gens, finales, lr0)
    raise ValueError(bras)
