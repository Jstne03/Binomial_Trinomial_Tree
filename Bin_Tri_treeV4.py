from __future__ import annotations
import math
import time
from datetime import date, timedelta
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

        # probabilité d'atteindre ce nœud depuis la racine (sert au pruning)
        self.proba = 0.0
        
    def forward(self):
        #retourne le prix du forward un pas plus loin
        return self.St * np.exp(self.tree.market.r * self.tree.deltaT)
    
    def up_price(self):
        #retourne le prix du noeud du haut un pas plus loin
        return self.forward() * self.tree.sqrt_alpha
    
    def down_price(self):
        #retourne le prix du noeud du bas un pas plus loin
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

        # probabilité d'atteindre ce nœud depuis la racine (sert au pruning)
        self.proba = 0.0
        

    def probs(self):
        if self.pu is None:
            return self.tree.pu, self.tree.pm, self.tree.pd
        return self.pu, self.pm, self.pd
    
    def forward(self):
        #retourne le prix du forward un pas plus loin
        return self.St * np.exp(self.tree.market.r * self.tree.deltaT)
    
    def up_price(self):
        #retourne le prix du noeud du haut un pas plus loin
        return self.forward() * self.tree.alpha
    
    def down_price(self):
        #retourne le prix du noeud du bas un pas plus loin
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

# Échelle des grecques en volatilité : résultats par point de vol (1 % = 0.01)
VEGA_SCALE = 0.01       # dV/dσ     -> /100
VANNA_SCALE = 0.01      # d²V/dSdσ  -> /100
VOMMA_SCALE = 0.0001    # d²V/dσ²   -> /10 000

#Black-Scholes pour un spot et une durée quelconques (sert au lissage de la dernière colonne)
def black_scholes_price(S, K, r, v, T, CP):
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if K <= 0:
        return S if CP == "C" else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    if CP == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


#Fonctions communes aux deux arbres

def _add_greek_nodes(tree, root, node_cls, factor):
    # ajoute un nœud au-dessus (S0*factor) et un en dessous (S0/factor) de la racine, à la
    # même date : on pricera ces 3 racines dans le même arbre (astuce du slide 17 pour delta/gamma)
    low = node_cls(tree, root.St / factor)
    up = node_cls(tree, root.St * factor)
    low.above, root.below = root, low
    root.above, up.below = up, root
    low.proba = up.proba = 1.0


def _delta_gamma(root):
    # delta et gamma à partir des 3 nœuds de la colonne 0 (valeurs déjà calculées par price())
    if root.above is None or root.below is None:
        raise ValueError("Arbre construit sans greek_nodes=True : pas de nœuds autour de la racine.")
    su, sm, sd = root.above.St, root.St, root.below.St
    vu, vm, vd = root.above.value, root.value, root.below.value
    delta = (vu - vd) / (su - sd)
    gamma = 2 * ((vu - vm) / (su - sm) - (vm - vd) / (sm - sd)) / (su - sd)
    return delta, gamma


def _count_nodes(last_trunk):
    # nombre total de nœuds, en remontant les colonnes par le tronc
    n = 0
    trunk = last_trunk
    while trunk is not None:
        node = trunk.lowest()
        while node is not None:
            n += 1
            node = node.above
        trunk = trunk.prec_trunk
    return n


#Arbre

class BinomialTree:
    def __init__(self, market, option, model, greek_nodes=False, smoothing=False, prune_threshold=None):
        self.smoothing = smoothing
        # prune_threshold : on coupe les nœuds dont la probabilité d'atteinte est < seuil
        # (ex. 1e-7) : ils n'ont alors qu'un seul fils avec probabilité 1 (celui qui les ramène
        # vers le centre de l'arbre, il n'y a pas de fils du milieu en binomial). None = pas de pruning.
        self.prune_threshold = prune_threshold or 0.0
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
        self.root.proba = 1.0
        self.last_trunk: TrunkBiNode | None = None
        if greek_nodes:
            _add_greek_nodes(self, self.root, BiNode, self.sqrt_alpha ** 2)

    def delta_gamma(self):
        return _delta_gamma(self.root)

    def count_nodes(self):
        return _count_nodes(self.last_trunk)
        
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
        Le nœud d'indice k (en partant du bas) de la colonne i a pour fils down l'indice k
        et pour fils up l'indice k+1 de la colonne i+1 (recombinaison).
        Pruning : un nœud élagué n'a qu'un fils, avec probabilité 1 : le fils down s'il est
        au-dessus des nœuds vivants, le fils up s'il est en dessous (donc vers le centre).
        """
        nodes = []
        node = trunk.lowest()
        while node is not None:
            nodes.append(node)
            node = node.above
        alive = [n.proba >= self.prune_threshold for n in nodes]
        lo = alive.index(True)
        hi = len(alive) - 1 - alive[::-1].index(True)
        top = len(nodes) - 1

        # indices de la colonne i+1 réellement nécessaires
        mn = 1 if lo > 0 else 0
        mx = max(hi + 1, top)

        # indice du tronc dans la colonne i, puis son fils dans la colonne i+1
        ti = next(k for k, n in enumerate(nodes) if n is trunk)
        if not alive[ti]:
            raise ValueError("Seuil de pruning trop élevé : le tronc est élagué")
        new_ti = ti + 1 if i % 2 == 0 else ti

        # création de la colonne i+1 (du bas vers le haut)
        new_nodes = []
        new_trunk = None
        for j in range(mn, mx + 1):
            price = nodes[j - 1].up_price() if j >= 1 else nodes[0].down_price()
            if j == new_ti:
                new_node = TrunkBiNode(self, price, prec_trunk=trunk, column=i + 1)
                new_trunk = new_node
            else:
                new_node = BiNode(self, price)
            if new_nodes:
                new_node.below = new_nodes[-1]
                new_nodes[-1].above = new_node
            new_nodes.append(new_node)

        # liens parent -> fils et probabilités d'atteinte
        for k, node in enumerate(nodes):
            if lo <= k <= hi:
                node.next_down = new_nodes[k - mn]
                node.next_up = new_nodes[k + 1 - mn]
                node.next_up.proba += self.u * node.proba
                node.next_down.proba += self.d * node.proba
            else:
                # nœud élagué : un seul fils, probabilité 1
                child = new_nodes[k - mn] if k > hi else new_nodes[k + 1 - mn]
                node.next_up = node.next_down = child
                child.proba += node.proba
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
        smooth = self.smoothing          # lissage : colonne N-1 valorisée par Black-Scholes sur Δt
        while trunk is not None:
            node = trunk.lowest()
            while node is not None:
                if smooth:
                    hold = black_scholes_price(node.St, opt.strike, self.market.r,
                                               self.market.vol, self.deltaT, opt.CP)
                else:
                    hold = self.discount * (self.u * node.next_up.value
                                            + self.d * node.next_down.value)
                if opt.is_american():
                    node.value = max(hold, opt.payoff(node.St))
                else:
                    node.value = hold
                node = node.above
            smooth = False
            trunk = trunk.prec_trunk
        return self.root.value
    
    
def price_binomial_tree(market, option, pricing_date, nb_step, prune_threshold=None, smoothing=False):
    model = Model(pricing_date, nb_step)
    return BinomialTree(market, option, model, prune_threshold=prune_threshold,
                        smoothing=smoothing).price()
    
class TrinomialTree:
    def __init__(self, market, option, model, greek_nodes=False, smoothing=False, prune_threshold=None):
        self.smoothing = smoothing
        # prune_threshold : on coupe les nœuds dont la probabilité d'atteinte est < seuil
        # (ex. 1e-7) : ils n'ont alors qu'un fils (le milieu) avec probabilité 1. None = pas de pruning.
        self.market = market
        self.option = option
        self.model = model
        self.prune_threshold = prune_threshold or 0.0
        self.deltaT = (option.t - model.pricing_date).days / 365 / model.nb_step
        
        e = np.exp((market.vol ** 2) * self.deltaT) 
        self.e = e
        self.alpha = np.exp(market.vol * np.sqrt(3 * self.deltaT))
        self.discount = np.exp(-market.r * self.deltaT)
        
        # les probas tant qu'il n'y a pas de div :
        self.pd = (e - 1) / ((1 - self.alpha)*(1 /self.alpha**2 - 1))
        self.pu = self.pd / self.alpha
        self.pm = 1 - self.pu - self.pd
        
        self.root = TrunkTriNode(self, market.s0, prec_trunk=None, column=0)
        self.root.proba = 1.0
        self.last_trunk: TrunkTriNode | None = None
        if greek_nodes:
            _add_greek_nodes(self, self.root, TriNode, self.alpha)
            
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
    
    def delta_gamma(self):
        return _delta_gamma(self.root)

    def count_nodes(self):
        return _count_nodes(self.last_trunk)

    def _alive(self, node):
        # un nœud est conservé (3 fils) si sa probabilité d'atteinte dépasse le seuil de pruning
        return node.proba >= self.prune_threshold

    def _propagate_proba(self, trunk):
        # proba des nœuds de la colonne suivante = somme (proba du parent x proba de transition)
        node = trunk.lowest()
        while node is not None:
            pu, pm, pd = node.probs()
            node.next_up.proba += pu * node.proba
            node.next_mid.proba += pm * node.proba
            node.next_down.proba += pd * node.proba
            node = node.above

    def calculate_column_div(self):
         # renvoie le numéro de colonne qui est impactée par le versement du div
         # calcul en entiers : le ceil sur des flottants décalait la colonne quand la date de
         # détachement tombe exactement sur une date de l'arbre
         days_total = (self.option.t - self.model.pricing_date).days
         days_div = (self.market.q_ex_div_date - self.model.pricing_date).days
         col = -((-days_div * self.model.nb_step) // days_total)
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
            new_trunk = self._build_div_column(trunk, i)
        else:
            new_trunk = self._build_normal_column(trunk, i)
        self._propagate_proba(trunk)
        return new_trunk
    
    def _build_normal_column(self, trunk: TrunkTriNode, i: int) -> TrunkTriNode:
        """
        Construit la colonne i+1 à partir de la colonne i, en montant la colonne i.
        Le fils du milieu d'un nœud est son forward ; le fils "up" d'un nœud est le fils
        du milieu du nœud du dessus (recombinaison). Un nœud élagué (pruning) n'a que son
        fils du milieu, avec probabilité 1. Le tronc de i+1 est le fils du milieu du tronc de i.
        """
        new_trunk = None
        prev_parent = None            # parent juste en dessous
        prev_mid = None               # son fils du milieu
        node = trunk.lowest()
        while node is not None:
            if node is trunk:
                mid = TrunkTriNode(self, node.forward(), prec_trunk=trunk, column=i + 1)
                new_trunk = mid
            else:
                mid = TriNode(self, node.forward())
            node.next_mid = mid
            if prev_mid is not None:
                mid.below = prev_mid
                prev_mid.above = mid
                if self._alive(prev_parent):
                    prev_parent.next_up = mid          # up du parent d'en dessous = ce mid
            if self._alive(node):
                node.next_down = prev_mid if prev_mid is not None else self._extend_below(mid)
            else:
                node.pu, node.pm, node.pd = 0.0, 1.0, 0.0
                node.next_down = node.next_up = mid     # branche monomiale
            prev_parent, prev_mid = node, mid
            node = node.above
        if self._alive(prev_parent):
            prev_parent.next_up = self._extend_above(prev_mid)
        return new_trunk

    def _extend_above(self, node):
        # crée (si besoin) le nœud juste au-dessus : même grille géométrique, raison alpha
        if node.above is None:
            up = TriNode(self, node.St * self.alpha)
            up.below = node
            node.above = up
        return node.above

    def _extend_below(self, node):
        # crée (si besoin) le nœud juste en dessous
        if node.below is None:
            down = TriNode(self, node.St / self.alpha)
            down.above = node
            node.below = down
        return node.below

    def _build_div_column(self, trunk, i):
        D = self.market.q_amount
        r_dt = np.exp(self.market.r * self.deltaT)
        sqrt_a = np.sqrt(self.alpha)

        # position du détachement dans le pas
        t_ex = (self.market.q_ex_div_date - self.model.pricing_date).days / 365
        delta = max(0.0, (i + 1) * self.deltaT - t_ex)   # t+Δt − t_ex
        D_fwd = D * np.exp(self.market.r * delta)        # D·e^{r(t+Δt−t_ex)}
        frac = delta / self.deltaT                       # (t+Δt−t_ex)/Δt

        new_trunk_price = trunk.St * r_dt - D_fwd
        if new_trunk_price <= 0:
            raise ValueError("Dividende trop grand : prix du tronc négatif")

        # nouveau tronc ; le reste de la grille (raison alpha) est créé à la demande
        new_trunk = TrunkTriNode(self, new_trunk_price, prec_trunk=trunk, column=i + 1)
        cursor = new_trunk          # nœud de la nouvelle colonne le plus proche du E courant

        # passe 1 : on monte la colonne i ; E croît avec le parent, donc le curseur
        # se déplace de proche en proche avec .above / .below
        node = trunk.lowest()
        while node is not None:
            E = node.forward() - D_fwd                      # S e^{rΔt} − D e^{r(...)}
            if E > 0:
                # nœud de la grille le plus proche de E (rapport < sqrt(alpha))
                while E > cursor.St * sqrt_a:
                    cursor = self._extend_above(cursor)
                while E < cursor.St / sqrt_a:
                    cursor = self._extend_below(cursor)
                node.next_mid = cursor
                if self._alive(node):
                    node.next_up = self._extend_above(cursor)
                    node.next_down = self._extend_below(cursor)
                    V = (node.St - frac * D) ** 2 * r_dt ** 2 * (self.e - 1)
                    try:
                        node.pu, node.pm, node.pd = self.probas(cursor.St, E, V)
                    except ValueError:
                        # noeud quasi mort : branche monomiale
                        node.pu, node.pm, node.pd = 0.0, 1.0, 0.0
                else:
                    # pruning : un seul fils (le milieu), probabilité 1
                    node.next_up = node.next_down = cursor
                    node.pu, node.pm, node.pd = 0.0, 1.0, 0.0
            node = node.above

        # passe 2 : parents "morts" (S < D) -> branche monomiale vers le nœud le plus bas
        lowest = new_trunk.lowest()
        node = trunk.lowest()
        while node is not None:
            if node.next_mid is None:
                node.pu, node.pm, node.pd = 0.0, 1.0, 0.0
                node.next_up = node.next_mid = node.next_down = lowest
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
        # lissage : colonne N-1 valorisée par Black-Scholes sur Δt (impossible si le dividende y tombe)
        smooth = self.smoothing and self.col_div != self.model.nb_step
        while trunk is not None:
            node = trunk.lowest()
            while node is not None:
                if smooth:
                    hold = black_scholes_price(node.St, opt.strike, self.market.r,
                                               self.market.vol, self.deltaT, opt.CP)
                else:
                    pu, pm, pd = node.probs()
                    hold = self.discount * (pu * node.next_up.value
                                            + pm * node.next_mid.value
                                            + pd * node.next_down.value)
                
                if opt.is_american():
                    node.value = max(hold, opt.payoff(node.St))
                else:
                    node.value = hold
                node = node.above
            smooth = False
            trunk = trunk.prec_trunk
        return self.root.value
    
    
def price_trinomial_tree(market, option, pricing_date, nb_step, prune_threshold=None, smoothing=False):
    model = Model(pricing_date, nb_step)
    return TrinomialTree(market, option, model, prune_threshold=prune_threshold,
                         smoothing=smoothing).price()


def price_and_greeks(market, option, pricing_date, nb_step, tree_type="tri",
                     vol_shift=0.01, prune_threshold=None, smoothing=False):
    """
    Prix + grecques à partir de l'arbre.
      - delta, gamma : lus sur 3 nœuds autour de la racine (un seul pricing).
      - vega, vomma  : différences finies sur la volatilité (pas d'astuce d'arbre pour eux, slide 17).
    tree_type : "tri" ou "bin".
    smoothing  : lisse la dernière colonne avec Black-Scholes -> supprime les oscillations de l'arbre
                 (utile surtout pour vega/vomma, qui sont des dérivées secondes bruitées sinon).
    """
    if tree_type == "tri":
        make = lambda m, g, d=pricing_date: TrinomialTree(m, option, Model(d, nb_step),
                                          greek_nodes=g, prune_threshold=prune_threshold,
                                          smoothing=smoothing)
    elif tree_type == "bin":
        make = lambda m, g, d=pricing_date: BinomialTree(m, option, Model(d, nb_step), greek_nodes=g,
                                     prune_threshold=prune_threshold, smoothing=smoothing)
    else:
        raise ValueError("tree_type doit valoir 'tri' ou 'bin'")

    tree = make(market, True)
    price = tree.price()
    delta, gamma = tree.delta_gamma()

    def price_at_vol(vol):
        # prix et delta (lu sur les nœuds autour de la racine) pour une autre volatilité
        m = Market(market.s0, market.r, vol, market.q_ex_div_date, market.q_amount)
        t = make(m, True)
        return t.price(), t.delta_gamma()[0]

    h = vol_shift
    (p_up, d_up), (p_mid, _), (p_dn, d_dn) = (price_at_vol(market.vol + h), price_at_vol(market.vol),
                                              price_at_vol(market.vol - h))
    # dérivées par unité de vol, puis mises à l'échelle "par point de vol" (1 % = 0.01)
    vega = (p_up - p_dn) / (2 * h) * VEGA_SCALE
    vomma = (p_up - 2 * p_mid + p_dn) / h ** 2 * VOMMA_SCALE
    vanna = (d_up - d_dn) / (2 * h) * VANNA_SCALE

    # theta : différence finie sur la date de pricing (1 jour), exprimé par an
    day = timedelta(days=1)
    ex = market.q_ex_div_date
    if (option.t - pricing_date).days <= 1:
        theta = None                      # moins d'un jour restant : n/a
    elif market.q_amount and ex is not None and pricing_date < ex <= pricing_date + day:
        # avancer d'un jour ferait sauter le dividende : on recule d'un jour à la place
        theta = (price - make(market, False, pricing_date - day).price()) * 365
    else:
        theta = (make(market, False, pricing_date + day).price() - price) * 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "vomma": vomma,"vanna": vanna, "theta": theta}
    


#Black-Scholes (qui servira de référence pour la convergence uniquement pr 
# les options européennes et sans dividende)
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

    def greeks(self):
        # grecques analytiques (vega et vomma par unité de volatilité, pas par point de %)
        T = (self.option.t - self.pricing_date).days / 365
        S, K, r, v = self.market.s0, self.option.strike, self.market.r, self.market.vol
        d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        delta = self.norm_cdf(d1) if self.option.CP == "C" else self.norm_cdf(d1) - 1
        gamma = pdf / (S * v * math.sqrt(T))
        vega = S * pdf * math.sqrt(T)
        vomma = vega * d1 * d2 / v * VOMMA_SCALE
        vanna = -pdf * d2 / v * VANNA_SCALE
        vega = vega * VEGA_SCALE
        
        decay = -S * pdf * v / (2 * math.sqrt(T))
        if self.option.CP == "C":
            theta = decay - r * K * math.exp(-r * T) * self.norm_cdf(d2)
        else:
            theta = decay + r * K * math.exp(-r * T) * self.norm_cdf(-d2)
            
        return {"delta": delta, "gamma": gamma, "vega": vega, "vomma": vomma,"vanna": vanna, "theta": theta}