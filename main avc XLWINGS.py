import xlwings as xw

from Bin_Tri_treeV4 import (
    Market,
    Option,
    Model,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_and_greeks,
)

def world():
    wb: xw.Book = xw.Book.caller()
    top: xw.Range = wb.sheets['Main'].range('TopCell')
    
    #Market
    St : xw.Range = wb.sheets['Main'].range('Stock_price')
    r : xw.Range = wb.sheets['Main'].range('Interest_rate')
    sigma : xw.Range = wb.sheets['Main'].range('volatility')
    ex_dividend_date : xw.Range = wb.sheets['Main'].range('Ex_dividend_date')
    q_amount : xw.Range = wb.sheets['Main'].range('Div_amount')
    
    #Model
    pricing_date : xw.Range = wb.sheets['Main'].range('Pricing_date')
    nb_step : xw.Range = wb.sheets['Main'].range('Nb_steps')
    
    #Tree
    prune_threshold : xw.Range = wb.sheets['Main'].range('Pruning')
    maturity : xw.Range = wb.sheets['Main'].range('Nb_nodes')
    smoothing : xw.Range = wb.sheets['Main'].range('Limites')
    grecques : xw.Range = wb.sheets['Main'].range('Grecques')
    
    #Option
    maturity_date : xw.Range = wb.sheets['Main'].range('Maturity')
    CP : xw.Range = wb.sheets['Main'].range('CP')
    Exercise : xw.Range = wb.sheets['Main'].range('Exercise')
    K : xw.Range = wb.sheets['Main'].range('Strike')
    
    
    BS_price : xw.Range = wb.sheets['Main'].range('BS_price')
    Tree_price : xw.Range = wb.sheets['Main'].range('Tree_price')
    
    market = Market(s0=St, r=r, vol=sigma, q_ex_div_date=ex_dividend_date, q_amount=q_amount)
    option = Option(strike=K, t=maturity_date, CP=CP, exercise=Exercise)
    model = Model(pricing_date, nb_step)
    
    #trinomial Tree
    tri_tree = TrinomialTree(market, option, model, prune_threshold=prune_threshold)
    tri_price = tri_tree.price()
    
    #
    firstRow: int = 0
    # look for the first empty cell in A column
    while len(str(top.offset(firstRow, 0).value)) > 5:
        firstRow += 1
    # Fill the cell with some calculations
    BS_price.value = 2
    Tree_price = 2
    top.offset(firstRow, 0).value = factorial(firstRow)


@xw.func
def factorial(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)


# example of UDF for test
@xw.func
def double_sum(x: float, y: float) -> float:
    """Returns twice the sum of the two arguments"""
    return 2 * (x + y)