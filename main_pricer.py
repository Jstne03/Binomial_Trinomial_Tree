import xlwings as xw
from datetime import date, datetime
from typing import Any, Dict, List
from dataclasses import dataclass

from Bin_Tri_treeV5 import (
    Market,
    Option,
    Model,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_and_greeks,
)

from tree_svg_viewer import (
    binomial_tree_to_svg,
    trinomial_tree_to_svg,
    SvgStyle
    )

from Convergence_BS_graphique import ConvergenceBS

def get_sheet(name: str = "Main") -> xw.Sheet:
    #Retourne la feuille `name` du classeur excel
    wb: xw.Book = xw.Book.caller()
    return wb.sheets[name]

def _to_date(x: Any) -> date:
    # Excel renvoie un datetime (ou None si la cellule est vide)
    if x is None:
        return None
    return x.date() if isinstance(x, datetime) else x

def _flag(x : Any) -> bool:
    # accepte TRUE/FALSE, 1/0, "oui"/"non"...
    return x is True or str(x).strip().lower() in ("1", "oui", "yes", "true", "vrai", "x")

def _val(ws: xw.Sheet, name: str) -> Any:
    #Lit la valeur de la plage nommée 'name'
    return ws.range(name).value

@dataclass(frozen=True)
class CommonInputs:
    """Entrées Excel communes aux trois fonctions."""
    market: Market
    option: Option
    pricing_date: date
    prune: float
    smoothing: bool
    has_div: bool

def _read_market(ws: xw.Sheet) -> Market:
    #Construit le Market à partir des cellules Excel
    return Market(
        _val(ws, "Stock_price"),
        _val(ws, "Interest_rate"),
        _val(ws, "Volatility"),
        _to_date(_val(ws, "Ex_dividend_date")),
        _val(ws, "Div_amount"),
    )

def _read_option(ws: xw.Sheet) -> Option:
    #Construit l'Option à partir des cellules Excel
    return Option(
        _val(ws, "Strike"),
        _to_date(_val(ws, "Maturity")),
        str(_val(ws, "CP")).strip().upper(),
        str(_val(ws, "Exercise")).strip().upper(),
        _flag(_val(ws, "Digital")),
        _val(ws, "Payout") or 1, #pour éviter les erreurs si la cellule est vide
        
    )

def read_inputs(ws: xw.Sheet) -> CommonInputs:
    #Lit toutes les entrées communes de la feuille
    has_div = (
        bool(_val(ws, "Div_amount"))
        and _to_date(_val(ws, "Ex_dividend_date")) is not None
    )
    return CommonInputs(
        market=_read_market(ws),
        option=_read_option(ws),
        pricing_date=_to_date(_val(ws, "Pricing_date")),
        prune=_val(ws, "Pruning") or None,
        smoothing=_flag(_val(ws, "Limites")),
        has_div=has_div,
    )


GREEK_KEYS = ["delta", "gamma", "vega", "vomma", "vanna", "theta"]   # ds même ordre que les libellés Excel

def _greeks_column(g : Dict[str, float]) -> List[List[float]]:
    # liste de listes = écriture verticale (une cellule par grecque)
    return [[g[k]] for k in GREEK_KEYS]

OUTPUT_NAMES = {
    "bs_price": "BS_price",
    "bin_price": "Bin_tree_price",
    "tri_price": "Trin_Tree_price",
    "nb_nodes": "Nb_nodes",
    "bs_greeks": "BS_grecques",
    "bin_greeks": "Bin_tree_grecques",
    "tri_greeks": "Trin_tree_grecques",
}

def _output_cells(ws: xw.Sheet) -> Dict[str, xw.Range]:
    """Retourne les cellules de sortie indexées par clé."""
    return {key: ws.range(name) for key, name in OUTPUT_NAMES.items()}

#----------------------- PRICER DES OPTIONS ----------------------------------
@xw.func
def excel_pricer() -> None:
    ws = get_sheet()
    ##------- INPUT
    inp = read_inputs(ws)
    nb_step = int(_val(ws, "Nb_steps"))
    want_greeks = _flag(_val(ws, "Grecques"))
    model = Model(inp.pricing_date, nb_step)
    ##------- OUTPUT (cellules, on les écrira à la fin)
    out = _output_cells(ws)

    try:
        # trinomial
        tri_tree = TrinomialTree(inp.market, inp.option, model,prune_threshold=inp.prune, smoothing=inp.smoothing)
        tri_price = tri_tree.price()
        out["nb_nodes"].value = tri_tree.count_nodes()
        tri_greeks = None
        if want_greeks:
            tri_greeks = price_and_greeks(inp.market, inp.option, inp.pricing_date, nb_step,
                                          tree_type="tri", prune_threshold=inp.prune,smoothing=inp.smoothing)

        # binomial : on ne le calcule pas s'il y a un dividende
        bin_price, bin_greeks = "impossible (dividende)", None
        if not inp.has_div:
            bin_price = BinomialTree(inp.market, inp.option, model,prune_threshold=inp.prune,smoothing=inp.smoothing).price()
            if want_greeks:
                bin_greeks = price_and_greeks(inp.market, inp.option, inp.pricing_date, nb_step,
                                              tree_type="bin", prune_threshold=inp.prune,smoothing=inp.smoothing)

        # Black-Scholes (sans dividende)
        bs = BlackScholes(inp.market, inp.option, inp.pricing_date)
        bs_price = bs.price_option()
        bs_greeks = bs.greeks() if want_greeks else None

    except ValueError as e:
        out["tri_price"].value = f"Erreur : {e}"
        return

    # None vide les cellules (grecques non demandées, ou binomial impossible)
    out["bs_price"].value = bs_price
    out["bin_price"].value = bin_price
    out["tri_price"].value = tri_price
    out["bs_greeks"].value = _greeks_column(bs_greeks) if bs_greeks else None
    out["bin_greeks"].value = _greeks_column(bin_greeks) if bin_greeks else None
    out["tri_greeks"].value = _greeks_column(tri_greeks) if tri_greeks else None
    
#----------------------- CREATION DES GRAPHIQUES -----------------------------    
@xw.func
def create_graphiques_arbres() -> None:
    ws = get_sheet()

    ##------- INPUT
    ws = get_sheet()
    inp = read_inputs(ws)
    model = Model(inp.pricing_date, int(_val(ws, "Nb_steps")))

    ##------- OUTPUT (cellules, on les écrira à la fin)
    visualisation_bin_cell = ws.range('Visualisation_bin')
    visualisation_trin_cell = ws.range('Visualisation_trin')
    
    bin_graph = "impossible (dividende)"
    try:
        if not inp.has_div:
            bin_tree = BinomialTree(inp.market, inp.option, model,prune_threshold=inp.prune, smoothing=inp.smoothing)
            bin_tree.price()
            bin_graph = binomial_tree_to_svg(bin_tree, "binomial_tree_visualisation.svg", SvgStyle(log_scale=True))
            
        # trinomial
        trin_tree = TrinomialTree(inp.market, inp.option, model,prune_threshold=inp.prune, smoothing=inp.smoothing)
        trin_tree.price()
        tri_graph = trinomial_tree_to_svg(trin_tree, "trinomial_tree_visualisation.svg", SvgStyle(log_scale=True))

    except ValueError as e:
        visualisation_bin_cell.value = f"Erreur : {e}"
        return

    visualisation_bin_cell.value = bin_graph
    visualisation_trin_cell.value = tri_graph

@xw.func
def create_graphiques_convergence() -> None:
    #pour étudier la convergence vers B&S on ne veut pas de dividende ni les grecques
    ws = get_sheet()

    ##------- INPUT
    ws = get_sheet()
    inp = read_inputs(ws)
    max_nb_pas = int(_val(ws, "CV_max_nb_pas"))

    ##------- OUTPUT (cellules, on les écrira à la fin)
    CV_BS_bin_cell = ws.range('CV_BS_bin')
    CV_BS_trin_cell = ws.range('CV_BS_trin')

    try:
        # trinomial
        tri_graph_class = ConvergenceBS(inp.market, inp.option, inp.pricing_date,'trinomial',
                                 prune_threshold=inp.prune, smoothing=inp.smoothing)
 
        tri_graph = tri_graph_class.convergence_black_scholes(max_nb_pas)

        #binomial
        bin_graph = None
        if not inp.has_div:
            bin_graph_class = ConvergenceBS(inp.market, inp.option, inp.pricing_date,'binomial',
                                     prune_threshold=inp.prune, smoothing=inp.smoothing)
            
            bin_graph = bin_graph_class.convergence_black_scholes(max_nb_pas)

    except ValueError as e:
        CV_BS_bin_cell.value = f"Erreur : {e}"
        return

    CV_BS_bin_cell.value = bin_graph
    CV_BS_trin_cell.value = tri_graph