from __future__ import annotations
import math
import time
from datetime import date
import numpy as np


#Données d'entrée
#les classes Market, Option et Model sont les mêmes pour les arbres binomiaux et trinomiaux.
#en revanche il existe deux types de "noeuds" l'un pr les binomiaux, l'autre pour les trinomiaux.
class Market:
    # classe qui initialise le prix du sous-jacent en date t=0,
    # le taux d'intérêt et la volatilité et le dividende
    def __init__(self, s0, r, vol,q_ex_div_date=None,q_amount=None):
        self.s0 = s0
        self.r = r
        self.vol = vol
        self.q_ex_div_date = q_ex_div_date
        self.q_amount = q_amount

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

class BiNode:
    # Un nœud = un prix du sous-jacent + des pointeurs vers d'autres nœuds.
    def __init__(self, tree: "BinomialTree", St: float):
        self.tree = tree
        self.St = St
        # colonne suivante
        
        self.next_up: BiNode | None = None
        self.next_down: BiNode | None = None
        # même colonne
        self.above: BiNode | None = None
        self.below: BiNode | None = None
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

class TrunkBiNode(BiNode):
    """Nœud du tronc : a en plus un lien vers le tronc de la colonne précédente."""
    def __init__(self, tree, St, prec_trunk: "TrunkBiNode | None" = None, column: int = 0):
        super().__init__(tree, St)
        self.prec_trunk = prec_trunk # lien arrière (pour le backward)
        self.column = column # numéro de la colonne (0 = racine)

class TriNode:
    # Un nœud = un prix du sous-jacent + des pointeurs vers d'autres nœuds.
    def __init__(self, tree: "TrinomialTree", St: float):
        self.tree = tree
        self.St = St
        # colonne suivante
        
        self.next_up: TriNode | None = None
        self.next_mid: TriNode | None = None
        self.next_down: TriNode | None = None
        # même colonne
        self.above: TriNode | None = None
        self.below: TriNode | None = None
        # valeur de l'option en ce nœud (remplie pendant le backward)
        self.value = 0.0
        
        #les probas sont les mêmes tant qu'il n'y a pas de dividende, 
        #elles changent à la colonne juste apres le dividende 
        #et dépendent du noeud 
        self.pu = None
        self.pm = None
        self.pd = None
        

    def probs(self):
        if self.pu is None:
            return self.tree.pu, self.tree.pm, self.tree.pd
        return self.pu, self.pm, self.pd
    
    def forward(self):
        #retourne le prix du forward un pas plus loin
        return self.St * np.exp(self.tree.market.r * self.tree.deltaT)
    
    def up_price(self):
        return self.forward() * self.tree.alpha
    
    def down_price(self):
        return self.forward() / self.tree.alpha
    
    def check_expected_value(self):
        # pu*S_up + pm*S_mid + pd*S_down doit être égal au forward
        pu, pm, pd = self.probs()
        return pu * self.up_price() + pm * self.forward() + pd * self.down_price()
    
    def lowest(self):
        #Descend jusqu'au nœud le plus bas de la colonne."""
        node = self
        while node.below is not None:
            node = node.below
        return node

class TrunkTriNode(TriNode):
    """Nœud du tronc : a en plus un lien vers le tronc de la colonne précédente."""
    def __init__(self, tree, St, prec_trunk: "TrunkTriNode | None" = None, column: int = 0):
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
        self.root = TrunkBiNode(self, market.s0, prec_trunk=None, column=0)
        self.last_trunk: TrunkBiNode | None = None
        
    # ---------- construction (forward) ----------
    def build(self):
        """Construit l'arbre colonne par colonne en suivant le tronc."""
        trunk = self.root
        # var pas de tps = cb de tps en jour
        for i in range(self.model.nb_step):
            
            trunk = self._build_next_column(trunk, i)
        self.last_trunk = trunk
        
    def _build_next_column(self, trunk: TrunkBiNode, i: int) -> TrunkBiNode:
        """
        Construit la colonne i+1 à partir de la colonne i (accessible via son tronc).
        Renvoie le tronc de la colonne i+1.
        Tronc : colonne paire -> nœud du milieu ; colonne impaire -> juste au-dessus.
        Donc le tronc alterne : up, down, up, down...
        """
        bottom = trunk.lowest()
        # nœud le plus bas de la nouvelle colonne
        new_bottom = BiNode(self, bottom.down_price())
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
                new_up = TrunkBiNode(self, price, prec_trunk=trunk, column=i + 1)
                new_trunk = new_up
            else:
                new_up = BiNode(self, price)
                    
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
    
    
def price_binomial_tree(market, option, pricing_date, nb_step):
    model = Model(pricing_date, nb_step)
    return BinomialTree(market, option, model).price()
    
class TrinomialTree:
    def __init__(self, market, option, model):
        self.market = market
        self.option = option
        self.model = model
        self.deltaT = (option.t - model.pricing_date).days / 365 / model.nb_step
        
        e = np.exp((market.vol ** 2) * self.deltaT) 
        self.e = e
        self.alpha = np.exp(market.vol * np.sqrt(3 * self.deltaT))
        self.discount = np.exp(-market.r * self.deltaT)
        
        # les probas tant qu'il n' a pas de div :
        self.pd = (e - 1) / ((1 - self.alpha)*(1 /self.alpha**2 - 1))
        self.pu = self.pd / self.alpha
        self.pm = 1 - self.pu - self.pd
        
        self.root = TrunkTriNode(self, market.s0, prec_trunk=None, column=0)
        self.last_trunk: TrunkTriNode | None = None
            
        # pour connaître la colonne qui est impactée par le dividende
        self.col_div = None
        if self.market.q_ex_div_date and self.market.q_amount:
            self.col_div = self.calculate_column_div()
        

    def probas(self, S_next_m, E, V):
        #calcule les proba aux noeuds
        a = self.alpha
        pd = ((V + E**2) / S_next_m**2 - 1 - (a + 1) * (E / S_next_m - 1)) \
             / ((1 - a) * (a**-2 - 1))
        pu = (E / S_next_m - 1 - (a**-1 - 1) * pd) / (a - 1)
        pm = 1 - pu - pd
        if min(pu, pm, pd) < 0:
            raise ValueError(f"Probabilités négatives : pu={pu}, pm={pm}, pd={pd}")
        return pu, pm, pd
    
    def calculate_column_div(self):
         # renvoie le numéro de colonne qui est impactée par le versement du div
         col = math.ceil((self.market.q_ex_div_date - self.model.pricing_date).days / 365 / self.deltaT)
         if col < 1 or col > self.model.nb_step:
             return None          # dividende hors de la vie de l'option
         return col

    
    # ---------- construction (forward) ----------
    def build(self):
        """Construit l'arbre colonne par colonne en suivant le tronc."""
        trunk = self.root
        for i in range(self.model.nb_step):
            trunk = self._build_next_column(trunk, i)
        self.last_trunk = trunk
     
    def _build_next_column(self, trunk, i):
        if self.col_div is not None and i + 1 == self.col_div:
            # colonne impactée par le dividende
            return self._build_div_column(trunk, i)
        return self._build_normal_column(trunk, i)
    
    def _build_normal_column(self, trunk: TrunkTriNode, i: int) -> TrunkTriNode:
        """
        Construit la colonne i+1 à partir de la colonne i.
        Le nœud k de la colonne i a pour fils les nœuds k (down), k+1 (mid), k+2 (up).
        Le tronc de la colonne i+1 est le fils du milieu du tronc de la colonne i.
        """
        
        bottom = trunk.lowest()
        lower = TriNode(self, bottom.down_price())      # tout en bas de la nouvelle colonne
        if bottom is trunk:                          # cas colonne 0 (un seul nœud)
            middle = TrunkTriNode(self, bottom.forward(), prec_trunk=trunk, column=i + 1)
            new_trunk = middle
        else:
            middle = TriNode(self, bottom.forward())
            new_trunk = None
        middle.below = lower
        lower.above = middle
        
        node = bottom                                # on remonte la colonne i
        while node is not None:
            # le "up" de ce nœud est le "mid" du nœud du dessus
            if node.above is trunk:
                upper = TrunkTriNode(self, node.up_price(), prec_trunk=trunk, column=i + 1)
                new_trunk = upper
            else:
                upper = TriNode(self, node.up_price())
            upper.below = middle
            middle.above = upper
            
            node.next_down = lower
            node.next_mid = middle
            node.next_up = upper
            
            lower, middle = middle, upper
            node = node.above
        return new_trunk
    
    def _build_div_column(self, trunk, i):
        D = self.market.q_amount
        a = self.alpha
        r_dt = np.exp(self.market.r * self.deltaT)

        # position du détachement dans le pas
        t_ex = (self.market.q_ex_div_date - self.model.pricing_date).days / 365
        delta = max(0.0, (i + 1) * self.deltaT - t_ex)   # t+Δt − t_ex
        D_fwd = D * np.exp(self.market.r * delta)        # D·e^{r(t+Δt−t_ex)}
        frac = delta / self.deltaT                       # (t+Δt−t_ex)/Δt

        new_trunk_price = trunk.St * r_dt - D_fwd
        if new_trunk_price <= 0:
            raise ValueError("Dividende trop grand : prix du tronc négatif")

        # nouvelle colonne : grille géométrique autour du nouveau tronc
        n = i + 1
        nodes, prev = {}, None
        for k in range(-n, n + 1):
            price = new_trunk_price * a ** k
            nd = (TrunkTriNode(self, price, prec_trunk=trunk, column=i + 1)
                  if k == 0 else TriNode(self, price))
            if prev is not None:
                nd.below = prev
                prev.above = nd
            nodes[k] = nd
            prev = nd

        # pour chaque parent : E, V, nœud du milieu, probabilités
        node = trunk.lowest()
        while node is not None:
            E = node.forward() - D_fwd                              # S e^{rΔt} − D e^{r(...)}
            if E <= 0:
                raise ValueError("Dividende trop grand : espérance négative")
            V = (node.St - frac * D) ** 2 * r_dt ** 2 * (self.e - 1)

            m = round(np.log(E / new_trunk_price) / np.log(a))      # nœud le plus proche de E
            m = max(-i, min(i, m))                                  # m−1 et m+1 doivent exister
            S_next_m = new_trunk_price * a ** m

            node.pu, node.pm, node.pd = self.probas(S_next_m, E, V)
            node.next_down = nodes[m - 1]
            node.next_mid = nodes[m]
            node.next_up = nodes[m + 1]
            node = node.above
        return nodes[0]
        
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
                
                pu, pm, pd = node.probs()
                hold = self.discount * (pu * node.next_up.value
                                        + pm * node.next_mid.value
                                        + pd * node.next_down.value)
                
                if opt.is_american():
                    node.value = max(hold, opt.payoff(node.St))
                else:
                    node.value = hold
                node = node.above
            trunk = trunk.prec_trunk
        return self.root.value
    
    
def price_trinomial_tree(market, option, pricing_date, nb_step):
    model = Model(pricing_date, nb_step)
    return TrinomialTree(market, option, model).price()
    


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
    
    