import xlwings as xw
from datetime import datetime

from Bin_Tri_treeV4 import (
    Market,
    Option,
    Model,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_and_greeks,
)

from Graphique_arbreV3 import(
    BinomialGraphique,
    TrinomialGraphique
    )

#----------------------- PRICER DES OPTIONS ----------------------------------
def _to_date(x):
    # Excel renvoie un datetime (ou None si la cellule est vide)
    if x is None:
        return None
    return x.date() if isinstance(x, datetime) else x


def _flag(x):
    # accepte TRUE/FALSE, 1/0, "oui"/"non"...
    return x is True or str(x).strip().lower() in ("1", "oui", "yes", "true", "vrai", "x")


GREEK_KEYS = ["delta", "gamma", "vega", "vomma", "vanna", "theta"]   # ds même ordre que les libellés Excel

def _greeks_column(g):
    # liste de listes = écriture verticale (une cellule par grecque)
    return [[g[k]] for k in GREEK_KEYS]

@xw.func
def excel_pricer():
    wb = xw.Book.caller()
    ws = wb.sheets['Main']

    ##------- INPUT (on lit .value)
    s0 = ws.range('Stock_price').value
    r = ws.range('Interest_rate').value
    sigma = ws.range('Volatility').value
    ex_div_date = _to_date(ws.range('Ex_dividend_date').value)
    q_amount = ws.range('Div_amount').value

    pricing_date = _to_date(ws.range('Pricing_date').value)
    nb_step = int(ws.range('Nb_steps').value)

    prune = ws.range('Pruning').value or None
    smoothing = _flag(ws.range('Limites').value)
    want_greeks = _flag(ws.range('Grecques').value)

    maturity = _to_date(ws.range('Maturity').value)
    CP = str(ws.range('CP').value).strip().upper()
    exercise = str(ws.range('Exercise').value).strip().upper()
    K = ws.range('Strike').value

    ##------- OUTPUT (cellules, on les écrira à la fin)
    bs_price_cell = ws.range('BS_price')
    bin_price_cell = ws.range('Bin_tree_price')
    tri_price_cell = ws.range('Trin_Tree_price')
    nb_nodes_cell = ws.range('Nb_nodes')
    bs_greeks_cell = ws.range('BS_grecques')
    bin_greeks_cell = ws.range('Bin_tree_grecques')
    tri_greeks_cell = ws.range('Trin_tree_grecques')

    ##------- COMPUTATION
    market = Market(s0, r, sigma, ex_div_date, q_amount)
    option = Option(K, maturity, CP, exercise)
    model = Model(pricing_date, nb_step)
    has_div = bool(q_amount) and ex_div_date is not None

    try:
        # trinomial
        tri_tree = TrinomialTree(market, option, model,
                                 prune_threshold=prune, smoothing=smoothing)
        tri_price = tri_tree.price()
        nb_nodes = tri_tree.count_nodes()
        tri_greeks = None
        if want_greeks:
            tri_greeks = price_and_greeks(market, option, pricing_date, nb_step,
                                          tree_type="tri", prune_threshold=prune,
                                          smoothing=smoothing)

        # binomial : on ne le calcule pas s'il y a un dividende
        bin_price, bin_greeks = "impossible (dividende)", None
        if not has_div:
            bin_price = BinomialTree(market, option, model,
                                     prune_threshold=prune,
                                     smoothing=smoothing).price()
            if want_greeks:
                bin_greeks = price_and_greeks(market, option, pricing_date, nb_step,
                                              tree_type="bin", prune_threshold=prune,
                                              smoothing=smoothing)

        # Black-Scholes (sans dividende)
        bs = BlackScholes(market, option, pricing_date)
        bs_price = bs.price_option()
        bs_greeks = bs.greeks() if want_greeks else None

    except ValueError as e:
        tri_price_cell.value = f"Erreur : {e}"
        return

    ##------- ÉCRITURE DANS EXCEL
    bs_price_cell.value = bs_price
    tri_price_cell.value = tri_price
    bin_price_cell.value = bin_price
    nb_nodes_cell.value = nb_nodes

    # None vide les cellules (grecques non demandées, ou binomial impossible)
    bs_greeks_cell.value = _greeks_column(bs_greeks) if bs_greeks else None
    tri_greeks_cell.value = _greeks_column(tri_greeks) if tri_greeks else None
    bin_greeks_cell.value = _greeks_column(bin_greeks) if bin_greeks else None
    
#----------------------- CREATION DES GRAPHIQUES -----------------------------    
@xw.func
def create_graphiques_arbres():
    wb = xw.Book.caller()
    ws = wb.sheets['Main']

    ##------- INPUT
    s0 = ws.range('Stock_price').value
    r = ws.range('Interest_rate').value
    sigma = ws.range('Volatility').value
    ex_div_date = _to_date(ws.range('Ex_dividend_date').value)
    q_amount = ws.range('Div_amount').value

    pricing_date = _to_date(ws.range('Pricing_date').value)
    nb_step = int(ws.range('Nb_steps').value)

    prune = ws.range('Pruning').value or None
    smoothing = _flag(ws.range('Limites').value)
    #want_greeks = _flag(ws.range('Grecques').value)

    maturity = _to_date(ws.range('Maturity').value)
    CP = str(ws.range('CP').value).strip().upper()
    exercise = str(ws.range('Exercise').value).strip().upper()
    K = ws.range('Strike').value

    ##------- OUTPUT (cellules, on les écrira à la fin)
    visualisation_bin_cell = ws.range('Visualisation_bin')
    visualisation_trin_cell = ws.range('Visualisation_trin')

    ##------- COMPUTATION
    market = Market(s0, r, sigma, ex_div_date, q_amount)
    option = Option(K, maturity, CP, exercise)
    has_div = bool(q_amount) and ex_div_date is not None
    
    bin_graph = "impossible (dividende)"
    try:
        if not has_div:
            bin_graph_class = BinomialGraphique(market, option, pricing_date,
                                     prune_threshold=prune, smoothing=smoothing)
            
            bin_graph = bin_graph_class.to_svg(nb_step)
            
        # trinomial
        tri_graph_class = TrinomialGraphique(market, option, pricing_date,
                                 prune_threshold=prune, smoothing=smoothing)
        
        #graphique de convergence vers BS
        tri_graph = tri_graph_class.to_svg(nb_step)

    except ValueError as e:
        visualisation_bin_cell.value = f"Erreur : {e}"
        return

    ##------- ÉCRITURE DANS EXCEL
    visualisation_bin_cell.value = bin_graph
    visualisation_trin_cell.value = tri_graph


@xw.func
def create_graphiques_convergence():
    #pour étudier la convergence vers B&S on ne veut pas de dividende ni les grecques
    wb = xw.Book.caller()
    ws = wb.sheets['Main']

    ##------- INPUT (on lit .value)
    s0 = ws.range('Stock_price').value
    r = ws.range('Interest_rate').value
    sigma = ws.range('Volatility').value
    ex_div_date = _to_date(ws.range('Ex_dividend_date').value)
    q_amount = ws.range('Div_amount').value

    pricing_date = _to_date(ws.range('Pricing_date').value)
    nb_step = int(ws.range('Nb_steps').value)
    
    #pour l'étude de la CV, cb de pas max pour les créations d'arbre :
    max_nb_pas = int(ws.range('CV_max_nb_pas').value)
    

    prune = ws.range('Pruning').value or None
    smoothing = _flag(ws.range('Limites').value)

    maturity = _to_date(ws.range('Maturity').value)
    CP = str(ws.range('CP').value).strip().upper()
    exercise = str(ws.range('Exercise').value).strip().upper()
    K = ws.range('Strike').value

    ##------- OUTPUT (cellules, on les écrira à la fin)
    CV_BS_bin_cell = ws.range('CV_BS_bin')
    CV_BS_trin_cell = ws.range('CV_BS_trin')

    ##------- COMPUTATION
    market = Market(s0, r, sigma, ex_div_date, q_amount)
    option = Option(K, maturity, CP, exercise)
    model = Model(pricing_date, nb_step)
    has_div = bool(q_amount) and ex_div_date is not None

    try:
        # trinomial
        tri_graph_class = TrinomialGraphique(market, option, pricing_date,
                                 prune_threshold=prune, smoothing=smoothing)
        
        #graphique de convergence vers BS
        tri_graph = tri_graph_class.convergence_black_scholes(max_nb_pas)

        #binomial
        bin_graph = None
        if not has_div:
            bin_graph_class = BinomialGraphique(market, option, pricing_date,prune_threshold=prune,
                                    smoothing=smoothing)
            
            #graphique de convergence vers BS
            bin_graph = bin_graph_class.convergence_black_scholes(max_nb_pas)

    except ValueError as e:
        CV_BS_bin_cell.value = f"Erreur : {e}"
        return

    ##------- ÉCRITURE DANS EXCEL
    CV_BS_bin_cell.value = bin_graph
    CV_BS_trin_cell.value = tri_graph


