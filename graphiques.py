from __future__ import annotations
from typing import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import math
import time
from datetime import date

# classes et fonctions de pricing du module principal
from Bin_Tri_treeV5 import (
    Market,
    Option,
    Model,
    BiNode,
    TriNode,
    TrunkBiNode,
    TrunkTriNode,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_binomial_tree,
    price_trinomial_tree
)
    
class Graphiques():

    # Associe le type d'arbre à sa fonction de pricing
    PRICERS = {
        "binomial": price_binomial_tree,
        "trinomial": price_trinomial_tree,
    }

    def __init__(self, market : Market, option : Option, pricing_date : date,
                 tree_type: str,
                 prune_threshold : float =None, smoothing : bool =False) -> None:

        if tree_type not in self.PRICERS:
            raise ValueError(
                f"tree_type doit être 'binomial' ou 'trinomial', reçu : '{tree_type}'"
            )

        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.tree_type = tree_type
        self.prune_threshold = prune_threshold
        self.smoothing = smoothing


# ---------- Temps de calcul avec / sans pruning ----------

def _mesurer_temps(price_func: Callable[..., float], market: Market, option: Option,
                   pricing_date: date, nb_pas_list: Sequence[int],
                   prune_threshold: float | None, smoothing: bool = False,
                   repeats: int = 3) -> tuple[list[float], list[float]]:
    """
    Mesure le temps de calcul (en secondes) de price_func pour chaque N de nb_pas_list,
    sans pruning puis avec pruning. On garde le meilleur de `repeats` essais pour limiter
    le bruit (garbage collector, autres processus...). Un calcul à blanc est fait avant
    pour ne pas compter le premier appel (imports, caches).
    """
    def chrono(n: int, prune: float | None) -> float:
        meilleur = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            price_func(market, option, pricing_date, n, prune, smoothing)
            meilleur = min(meilleur, time.perf_counter() - t0)
        return meilleur

    price_func(market, option, pricing_date, 10, None, smoothing)   # échauffement

    # un temps par N, sans puis avec pruning
    sans = [chrono(n, None) for n in nb_pas_list]
    avec = [chrono(n, prune_threshold) for n in nb_pas_list]
    return sans, avec

def _tracer_temps(ax: Axes, nb_pas_list: Sequence[int], sans: list[float], avec: list[float],
                  prune_threshold: float, titre: str) -> None:
    """Trace les deux courbes (sans / avec pruning) en échelle log-log sur l'axe ax."""
    # une courbe par cas (sans / avec pruning)
    ax.plot(nb_pas_list, sans, marker="o", label="sans pruning")
    ax.plot(nb_pas_list, avec, marker="o", label=f"pruning {prune_threshold:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # légendes et titre de l'axe
    ax.set_xlabel("Nombre de pas N")
    ax.set_ylabel("Temps de calcul (s)")
    ax.set_title(titre)
    ax.grid(True, which="both")
    ax.legend()


def temps_calcul_bin_tri(market: Market, option: Option, pricing_date: date,
                         nb_pas_list: Sequence[int] = (50, 100, 200, 400, 700, 1000),
                         prune_threshold: float = 1e-7, smoothing: bool = False,
                         repeats: int = 3,
                         filename: str = "temps_calcul_bin_tri.png") -> str:
    """
    Une seule figure, deux graphiques côte à côte (binomial | trinomial) :
    temps de calcul avec et sans pruning en fonction du nombre de pas N.
    Le binomial ignore le dividende discret : à utiliser sans dividende pour comparer.
    """
    # temps de calcul de chaque arbre
    bin_sans, bin_avec = _mesurer_temps(price_binomial_tree, market, option, pricing_date,
                                        nb_pas_list, prune_threshold, smoothing, repeats)
    tri_sans, tri_avec = _mesurer_temps(price_trinomial_tree, market, option, pricing_date,
                                        nb_pas_list, prune_threshold, smoothing, repeats)

    # une figure, un graphique par arbre
    fig, (ax_bin, ax_tri) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    _tracer_temps(ax_bin, nb_pas_list, bin_sans, bin_avec, prune_threshold,
                  "Temps de calcul de l'arbre binomial")
    _tracer_temps(ax_tri, nb_pas_list, tri_sans, tri_avec, prune_threshold,
                  "Temps de calcul de l'arbre trinomial")
    # sauvegarde de la figure
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return f"figure sauvée dans '{filename}'"

# Etude de l'erreur de l'arbre en fonction de la valeur du strike

def erreur_vs_strike(market: Market, option: Option, pricing_date: date, nb_pas: int = 1000,
                     strikes: Sequence[float] | np.ndarray | None = None,
                     prune_threshold: float | None = None, smoothing: bool = False,
                     filename: str = "erreur_vs_strike_bin_tri.png") -> str:
    """
    Erreur de l'arbre (prix arbre - prix Black-Scholes) en fonction du strike K,
    pour un nombre de pas nb_pas fixé, binomial et trinomial.

    Si les deux erreurs ont le même ordre de grandeur, elles sont tracées sur un seul
    graphique ; si l'une écrase l'autre (rapport des erreurs max > 10), on trace deux
    graphiques côte à côte, chacun avec sa propre échelle.
    Par défaut strikes = 70 % à 130 % de S0. Option européenne sans dividende uniquement.
    """
    if option.exercise != "EU" or market.q_ex_div_date is not None:
        raise ValueError("Black-Scholes n'est la référence que pour un européen sans dividende.")

    if strikes is None:
        strikes = np.linspace(0.7 * market.s0, 1.3 * market.s0, 121)
    strikes = np.array(strikes, dtype=float)

    # erreur de chaque arbre (prix arbre - prix BS) pour chaque strike
    err_bin, err_tri = [], []
    for K in strikes:
        opt = Option(strike=float(K), t=option.t, CP=option.CP, exercise=option.exercise,
                     digital=option.digital, payout=option.payout)
        bs = BlackScholes(market, opt, pricing_date).price_option()
        err_bin.append(price_binomial_tree(market, opt, pricing_date, nb_pas, None, smoothing) - bs)
        err_tri.append(price_trinomial_tree(market, opt, pricing_date, nb_pas, prune_threshold, smoothing) - bs)
    err_bin, err_tri = np.array(err_bin), np.array(err_tri)

    # même échelle si les erreurs max des deux arbres sont du même ordre de grandeur
    m_bin, m_tri = np.max(np.abs(err_bin)), np.max(np.abs(err_tri))
    meme_echelle = max(m_bin, m_tri) <= 10 * max(min(m_bin, m_tri), 1e-300)

    titre = f"Erreur de l'arbre en fonction du strike (N = {nb_pas})"
    ylabel = r"$V_N - V_{BS}$"

    # cas 1 : un seul graphique avec les deux arbres
    if meme_echelle:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(strikes, err_bin, label="Binomial", color="tab:blue")
        ax.plot(strikes, err_tri, label="Trinomial", color="tab:orange")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(market.s0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Strike K")
        ax.set_ylabel(ylabel)
        ax.set_title(titre)
        ax.grid(True)
        ax.legend()
    # cas 2 : deux graphiques côte à côte, chacun avec sa propre échelle
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for ax, name, err, couleur in zip(axes, ("Binomial", "Trinomial"),
                                          (err_bin, err_tri), ("tab:blue", "tab:orange")):
            ax.plot(strikes, err, color=couleur)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axvline(market.s0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Strike K")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}")
            ax.grid(True)
        fig.suptitle(titre)

    # sauvegarde de la figure
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return f"figure sauvée dans '{filename}'" + ("" if meme_echelle else " (deux graphiques : échelles trop différentes)")