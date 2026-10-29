import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import date

from Bin_Tri_treeV4 import (
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

    def convergence_black_scholes(self, max_nb_pas: int, filename: str = None) -> str:
        # Affiche la convergence du prix de l'arbre vers le prix Black-Scholes.
        price_tree = self.PRICERS[self.tree_type]
        price_bs = BlackScholes(self.market, self.option, self.pricing_date).price_option()
    
        N_values = np.arange(10, max_nb_pas)
        errors = [
            n * (price_tree(self.market, self.option, self.pricing_date, n,
                            self.prune_threshold, self.smoothing) - price_bs)
            for n in N_values
        ]
    
        plt.figure(figsize=(10, 6))
        plt.scatter(N_values, errors, label=r"$N\,(V_N - V_{BS})$")
        plt.xlabel("Nombre de pas N")
        plt.ylabel(r"$N\,(V_N - V_{BS})$")
        plt.title(f"Convergence de l'arbre {self.tree_type} vers Black-Scholes")
        plt.grid(True)
        plt.legend()
    
        filename = filename or f"CV_BS_{self.tree_type}.png"
        plt.savefig(filename)
        return f"figure sauvée dans '{filename}'"