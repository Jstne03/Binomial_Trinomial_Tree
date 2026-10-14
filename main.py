from datetime import date

from Bin_Tri_treeV2 import (
    Market,
    Option,
    Model,
    BinomialTree,
    TrinomialTree
)


from Graphique_arbreV2 import (
    BinomialGraphique,
    TrinomialGraphique
)


if __name__ == "__main__":

    # Inputs utilisateur
    S0 = float(input("Prix initial du sous-jacent S0 : "))
    K = float(input("Strike K : "))
    r = float(input("Taux sans risque r (ex: 0.03 pour 3%) : "))
    sigma = float(input("Volatilité sigma (ex: 0.20 pour 20%) : "))

    pricing_date_input = input("Date de pricing (YYYY-MM-DD) : ")
    maturity_date_input = input("Date de maturité (YYYY-MM-DD) : ")
    dividend_date_input = input("Date de détahement de dividende (YYYY-MM-DD) : ")

    pricing_date = date.fromisoformat(pricing_date_input)
    maturity_date = date.fromisoformat(maturity_date_input)
    dividend_date = date.fromisoformat(dividend_date_input)
    
    montant_div = float(input("montant du dividende discret :"))
    
    cp = input("Type d'option (C pour Call, P pour Put) : ").upper()

    # Création du marché et de l'option
    market = Market(s0=S0, r=r, vol=sigma, q_ex_div_date=dividend_date, q_amount=montant_div)
    option = Option(
        strike=K,
        t=maturity_date,
        CP=cp,
        exercise="EU"
    )


    # # Données
    # pricing_date = date(2026, 9, 15)

    # market = Market(
    #     s0=100,
    #     r=0.03,
    #     vol=0.20
    # )

    # option = Option(
    #     strike=101,
    #     t=date(2027, 9, 2),
    #     CP="C",
    #     exercise="EU"
    # )

    # Création de l'arbre binomial
    bi_tree = BinomialTree(
        market,
        option,
        Model(
            pricing_date,
            5
        )
    )

    # Création de l'arbre trinomial
    tri_tree = TrinomialTree(
        market,
        option,
        Model(
            pricing_date,
            5
        )
    )

    # Calcul du prix
    bi_price = bi_tree.price()
    tri_price = tri_tree.price()

    print("Prix de l'option (arbre binomial):", bi_price)
    print("Prix de l'option (arbre trinomial):", tri_price)

    # Création de l'objet Graphique
    bi_graphique = BinomialGraphique(bi_tree)
    tri_graphique = TrinomialGraphique(tri_tree)

    # Graphique de convergence
    bi_graphique.convergence_black_scholes()
    tri_graphique.convergence_black_scholes()

    # Représentation de l'arbre
    bi_graphique.to_svg("arbre binomial.svg")
    tri_graphique.to_svg("arbre trinomial.svg")
    
