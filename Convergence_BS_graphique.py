from __future__ import annotations
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import date

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
    
class ConvergenceBS():

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
    
    def _speed(self, n_values: np.ndarray) -> np.ndarray:
        #Facteur d'échelle : sqrt(N) pour une digitale, N sinon
        n = n_values.astype(float)
        return np.sqrt(n) if self.option.digital else n

    def _label(self) -> str:
        #titre de la quantité tracée (pour la légende du graphique)
        factor = r"\sqrt{N}" if self.option.digital else "N"
        return rf"${factor}\,(V_N - V_{{BS}})$"

    def convergence_black_scholes(self, max_nb_pas: int,filename: str = None) -> str:
        #Trace la convergence de l'arbre vers Black-Scholes (sans dividende)
        price_tree = self.PRICERS[self.tree_type]
        price_bs = BlackScholes(self.market, self.option, self.pricing_date).price_option()

        # écart arbre - BS pour chaque nombre de pas, mis à l'échelle de la vitesse théorique
        n_values = np.arange(10, max_nb_pas)
        diffs = np.array([price_tree(self.market, self.option, self.pricing_date,int(n), self.prune_threshold, self.smoothing)
                          - price_bs for n in n_values])
        values = self._speed(n_values) * diffs

        # nuage de points et mise en forme
        plt.figure(figsize=(10, 6))
        plt.scatter(n_values, values, label=self._label())
        plt.xlabel("Nombre de pas N")
        plt.ylabel(self._label())
        plt.title(f"Convergence de l'arbre {self.tree_type} vers Black-Scholes")
        plt.grid(True)
        plt.legend()

        # sauvegarde de la figure
        filename = filename or f"CV_BS_{self.tree_type}.png"
        plt.savefig(filename)
        return f"figure sauvée dans '{filename}'"

    @staticmethod
    def etude_ordre_convergence(market: Market, option: Option, pricing_date: date,
                                nb_pas_list: Sequence[int] = tuple(range(10, 301, 10)),
                                prune_threshold: float | None = None, smoothing: bool = True,
                                filename: str = "ordre_convergence_bin_tri.png") -> str:
  
        # Black-Scholes n'est la référence que pour un européen sans dividende
        if option.exercise != "EU" or market.q_ex_div_date is not None:
            raise ValueError("Black-Scholes n'est la référence que pour un européen sans dividende.")

        # prix de référence BS et nombres de pas testés
        price_bs = BlackScholes(market, option, pricing_date).price_option()
        N = np.array(nb_pas_list, dtype=float)

        # écart arbre - BS pour chaque nombre de pas
        gaps = {
            "Binomial": np.array([price_binomial_tree(market, option, pricing_date, int(n), None, smoothing) - price_bs
                                for n in nb_pas_list]),
        }
        couleurs = {"Binomial": "tab:blue"}

        # facteurs d'échelle testés pour trouver l'ordre de convergence
        facteurs = [
            (np.sqrt(N), r"$\sqrt{N}\,(V_N - V_{BS})$"),
            (N, r"$N\,(V_N - V_{BS})$"),
            (N ** 2, r"$N^2\,(V_N - V_{BS})$"),
        ]

        # 1 ligne (binomial) x 3 colonnes (sqrt(N), N, N^2), échelle propre à chaque graphique
        fig, axes = plt.subplots(1, 3, figsize=(18, 9))

        # un graphique par facteur d'échelle
        for ligne, (name, gap) in enumerate(gaps.items()):
            for col, (facteur, titre) in enumerate(facteurs):
                ax = axes[col]
                ax.plot(N, facteur * gap, color=couleurs[name])
                ax.axhline(0, color="grey", linewidth=0.8)
                ax.set_title(f"{name} : {titre}")
                ax.set_xlabel("Nombre de pas N")
                ax.set_ylabel(titre)
                ax.grid(True)

        # titre global et sauvegarde de la figure
        fig.suptitle("Ordre de convergence vers Black-Scholes : arbre binomial")
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        return f"figure sauvée dans '{filename}'"