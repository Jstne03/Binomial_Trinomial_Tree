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

def pricer():
    wb: xw.Book = xw.Book.caller()
    
    ##-------INPUT
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
    maturity_date : xw.Range = wb.sheets['Main'].range('Nb_nodes')
    smoothing : xw.Range = wb.sheets['Main'].range('Limites')
    grecques : xw.Range = wb.sheets['Main'].range('Grecques')
    
    #Option
    maturity_date : xw.Range = wb.sheets['Main'].range('Maturity')
    CP : xw.Range = wb.sheets['Main'].range('CP')
    Exercise : xw.Range = wb.sheets['Main'].range('Exercise')
    K : xw.Range = wb.sheets['Main'].range('Strike')
    
    ##-------OUTPUT
    bs_price : xw.Range = wb.sheets['Main'].range('BS_price')
    bin_price : xw.Range = wb.sheets['Main'].range('Bin_tree_price')
    trin_price : xw.Range = wb.sheets['Main'].range('Trin_Tree_price')
    
    bs_grecques : xw.Range = wb.sheets['Main'].range('BS_grecques')
    bin_tree_grecques : xw.Range = wb.sheets['Main'].range('Bin_tree_grecques')
    trin_tree_grecques : xw.Range = wb.sheets['Main'].range('Trin_tree_grecques')
    
    ##--------COMPUTATION
    market = Market(s0=St, r=r, vol=sigma, q_ex_div_date=ex_dividend_date, q_amount=q_amount)
    option = Option(strike=K, t=maturity_date, CP=CP, exercise=Exercise)
    model = Model(pricing_date, nb_step)
    
    #trinomial Tree
    tri_tree = TrinomialTree(market, option, model, greek_nodes= grecques, prune_threshold=prune_threshold, smoothing=smoothing)
    tri_price = tri_tree.price()
    #rajouter les grecques 
    
    #binomial Tree
    #check before calling it if there's a dividend bc we can't do it if there is one so return None
    #rajouter les grecques 
    
    #Black&Scholes
    #rajouter les grecques
    
    #output in the excel file
    #d'abord les prix
    bs_price.value = 2
    trin_price.value = tri_price
    bin_price.value = 2
    
    #ensuite les grecques (attention ce sont des places de valeurs !)
    bs_grecques.value = 0
    bin_tree_grecques.value = 0 
    trin_tree_grecques.value = 0
    


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