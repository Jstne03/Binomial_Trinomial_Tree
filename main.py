from datetime import date

from binomialTreeV3 import (
    Market,
    Option,
    Model,
    BinomialTree,
)

from GraphiqueBinomial import Graphique

from TestUnitaireBinomialTree import TestBinomialTree


if __name__ == "__main__":

    # Inputs utilisateur
    S0 = float(input("Prix initial du sous-jacent S0 : "))
    K = float(input("Strike K : "))
    r = float(input("Taux sans risque r (ex: 0.03 pour 3%) : "))
    sigma = float(input("Volatilité sigma (ex: 0.20 pour 20%) : "))

    pricing_date_input = input("Date de pricing (YYYY-MM-DD) : ")
    maturity_date_input = input("Date de maturité (YYYY-MM-DD) : ")

    pricing_date = date.fromisoformat(pricing_date_input)
    maturity_date = date.fromisoformat(maturity_date_input)

    cp = input("Type d'option (C pour Call, P pour Put) : ").upper()

    # Création du marché et de l'option
    market = Market(s0=S0, r=r, vol=sigma)
    option = Option(
        strike=K,
        t=maturity_date,
        CP=cp,
        exercise="EU"
    )


    # Données
    pricing_date = date(2026, 9, 15)

    market = Market(
        s0=100,
        r=0.03,
        vol=0.20
    )

    option = Option(
        strike=101,
        t=date(2027, 9, 2),
        CP="C",
        exercise="EU"
    )

    # Création de l'arbre
    tree = BinomialTree(
        market,
        option,
        Model(
            pricing_date,
            5
        )
    )

    # Calcul du prix
    price = tree.price()

    print("Prix de l'option :", price)

    # Création de l'objet Graphique
    graphique = Graphique(tree)

    # Graphique de convergence
    graphique.convergence_black_scholes()

    # Représentation de l'arbre
    graphique.to_svg("mon_arbre.svg")