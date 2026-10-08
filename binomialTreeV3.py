from __future__ import annotations
import math
import time
from datetime import date
import numpy as np


#Données d'entrée

class Market:
    # classe qui initialise le prix du sous-jacent en date t=0, le taux d'intérêt et la volatilité
    def __init__(self, s0, r, vol):
        self.s0 = s0
        self.r = r
        self.vol = vol

class Option:
    def __init__(self, strike, t, CP, exercise):
        self.strike = strike
        self.t = t
        self.CP = CP
        self.exercise = exercise
        
    def payoff(self, s):
        # Valeur de l'exercice immédiat au prix s
        if self.CP == "C":
            return max(s - self.strike, 0.0)
        
        elif self.CP == "P":
            return max(self.strike - s, 0.0)
        else:
            raise ValueError
            
    
    def is_american(self):
        return self.exercise == "AM"

class Model:
    def __init__(self, pricing_date, nb_step):
        self.pricing_date = pricing_date
        self.nb_step = nb_step


#Nœuds

class Node:
    # Un nœud = un prix du sous-jacent + des pointeurs vers d'autres nœuds.
    def __init__(self, tree: "BinomialTree", St: float):
        self.tree = tree
        self.St = St
        # colonne suivante
        
        self.next_up: Node | None = None
        self.next_down: Node | None = None
        # même colonne
        self.above: Node | None = None
        self.below: Node | None = None
        # valeur de l'option en ce nœud (remplie pendant le backward)
        self.value = 0.0
        
    def forward(self):
        #retourne le prix du forward un pas plus loin
        return self.St * np.exp(self.tree.market.r * self.tree.deltaT)
    
    def up_price(self):
        return self.forward() * self.tree.sqrt_alpha
    
    def down_price(self):
        return self.forward() / self.tree.sqrt_alpha
    
    def check_expected_value(self):
        #uS_up + dS_down doit être égal au forward (test de cohérence).
        t = self.tree
        return t.u * self.up_price() + t.d * self.down_price()
    
    def lowest(self):
        #Descend jusqu'au nœud le plus bas de la colonne."""
        node = self
        while node.below is not None:
            node = node.below
        return node

class TrunkNode(Node):
    """Nœud du tronc : a en plus un lien vers le tronc de la colonne précédente."""
    def __init__(self, tree, St, prec_trunk: "TrunkNode | None" = None, column: int = 0):
        super().__init__(tree, St)
        self.prec_trunk = prec_trunk # lien arrière (pour le backward)
        self.column = column # numéro de la colonne (0 = racine)


#Arbre

class BinomialTree:
    def __init__(self, market, option, model):
        self.market = market
        self.option = option
        self.model = model
        self.deltaT = (option.t - model.pricing_date).days / 365 / model.nb_step
        e = np.exp((market.vol ** 2) * self.deltaT)
        self.sqrt_alpha = (1 + e) / 2 + np.sqrt(((1 + e) / 2) ** 2 - 1)
        self.discount = np.exp(-market.r * self.deltaT)
        self.u = 1 / (1 + self.sqrt_alpha)
        self.d = 1 - self.u
        self.root = TrunkNode(self, market.s0, prec_trunk=None, column=0)
        self.last_trunk: TrunkNode | None = None
        
    # ---------- construction (forward) ----------
    def build(self):
        """Construit l'arbre colonne par colonne en suivant le tronc."""
        trunk = self.root
        for i in range(self.model.nb_step):
            trunk = self._build_next_column(trunk, i)
        self.last_trunk = trunk
        
    def _build_next_column(self, trunk: TrunkNode, i: int) -> TrunkNode:
        """
        Construit la colonne i+1 à partir de la colonne i (accessible via son tronc).
        Renvoie le tronc de la colonne i+1.
        Tronc : colonne paire -> nœud du milieu ; colonne impaire -> juste au-dessus.
        Donc le tronc alterne : up, down, up, down...
        """
        bottom = trunk.lowest()
        # nœud le plus bas de la nouvelle colonne
        new_bottom = Node(self, bottom.down_price())
        bottom.next_down = new_bottom
        new_trunk = None
        node = bottom # on remonte la colonne i du bas vers le haut
        below_new = new_bottom # dernier nœud créé dans la colonne i+1
        while node is not None:
            # le fils "up" de ce nœud est-il le prochain nœud du tronc ?
            if i % 2 == 0:
                is_trunk_child = node is trunk
            else:
                is_trunk_child = node.above is trunk
                    
            price = node.up_price()
            if is_trunk_child:
                new_up = TrunkNode(self, price, prec_trunk=trunk, column=i + 1)
                new_trunk = new_up
            else:
                new_up = Node(self, price)
                    
            node.next_up = new_up
                
            # liens dans la nouvelle colonne
            new_up.below = below_new
            below_new.above = new_up
            # recombinaison : le nœud du dessus descend vers ce même nœud
            if node.above is not None:
                node.above.next_down = new_up
                    
            below_new = new_up
            node = node.above
        return new_trunk
    
    

        
    # ---------- pricing (backward) ----------
    def price(self):
        """Construit l'arbre puis calcule le prix de l'option en remontant."""
        self.build()
        opt = self.option
        # 1) maturité : valeur = payoff
        node = self.last_trunk.lowest()
        while node is not None:
            node.value = opt.payoff(node.St)
            node = node.above
        # 2) on remonte de tronc en tronc jusqu'à la racine
        trunk = self.last_trunk.prec_trunk
        while trunk is not None:
            node = trunk.lowest()
            while node is not None:
                hold = self.discount * (self.u * node.next_up.value
                                    + self.d * node.next_down.value)
                if opt.is_american():
                    node.value = max(hold, opt.payoff(node.St))
                else:
                    node.value = hold
                node = node.above
            trunk = trunk.prec_trunk
        return self.root.value
    
    
def price_tree(market, option, pricing_date, nb_step):
    model = Model(pricing_date, nb_step)
    return BinomialTree(market, option, model).price()
    


#Black-Scholes (qui servira de référence pour la convergence)
class BlackScholes():
    
    def __init__(self,market, option, pricing_date):
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        
    
    def norm_cdf(self,x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def price_option(self):
        T = (self.option.t - self.pricing_date).days / 365
        S, K, r, v = self.market.s0, self.option.strike, self.market.r, self.market.vol
        d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        if self.option.CP == "C":
            return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        #sinon c'est un put
        return K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
    
    