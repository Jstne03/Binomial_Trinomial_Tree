from datetime import date

from Bin_Tri_treeV2 import (
    Market,
    Option,
    Model,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_and_greeks,
)

from Graphique_arbre import (
    BinomialGraphique,
    TrinomialGraphique
)


def ask_float(msg, default=None):
    txt = input(msg).strip()
    return default if txt == "" else float(txt)


def show_greeks(name, g, ref=None):
    print(f"\nGrecques ({name}, delta/gamma lus sur l'arbre, vega/vomma par différences finies sur sigma) :")
    for k in ("delta", "gamma", "vega", "vomma"):
        line = f"  {k:6s} = {g[k]:.6f}"
        if ref is not None:
            line += f"   (Black-Scholes : {ref[k]:.6f})"
        print(line)


if __name__ == "__main__":

    # ---------- Inputs utilisateur ----------
    S0 = float(input("Prix initial du sous-jacent S0 : "))
    K = float(input("Strike K : "))
    r = float(input("Taux sans risque r (ex: 0.03 pour 3%) : "))
    sigma = float(input("Volatilité sigma (ex: 0.20 pour 20%) : "))

    pricing_date = date.fromisoformat(input("Date de pricing (YYYY-MM-DD) : "))
    maturity_date = date.fromisoformat(input("Date de maturité (YYYY-MM-DD) : "))

    div_txt = input("Date de détachement du dividende (YYYY-MM-DD, vide = pas de dividende) : ").strip()
    dividend_date, montant_div = None, None
    if div_txt:
        dividend_date = date.fromisoformat(div_txt)
        montant_div = float(input("Montant du dividende discret : "))

    cp = input("Type d'option (C pour Call, P pour Put) : ").upper()
    exercise = input("Exercice (EU pour européen, AM pour américain) : ").upper()
    nb_step = int(input("Nombre de pas N (ex: 200) : "))
    prune = ask_float("Seuil de pruning trinomial (ex: 1e-7, vide = pas de pruning) : ")

    market = Market(s0=S0, r=r, vol=sigma, q_ex_div_date=dividend_date, q_amount=montant_div)
    option = Option(strike=K, t=maturity_date, CP=cp, exercise=exercise)
    model = Model(pricing_date, nb_step)

    # ---------- Prix ----------
    tri_tree = TrinomialTree(market, option, model, prune_threshold=prune)
    tri_price = tri_tree.price()
    print("\nPrix de l'option (arbre trinomial):", tri_price)
    print("Nombre de nœuds de l'arbre trinomial :", tri_tree.count_nodes())

    has_div = dividend_date is not None
    if has_div:
        print("Arbre binomial non calculé : il ne gère pas le dividende discret (utiliser le trinomial).")
    else:
        bi_price = BinomialTree(market, option, model).price()
        print("Prix de l'option (arbre binomial):", bi_price)
        if exercise == "EU":
            print("Prix Black-Scholes :", BlackScholes(market, option, pricing_date).price_option())

    # ---------- Grecques à partir de l'arbre ----------
    # smoothing=True : dernière colonne valorisée par Black-Scholes, sinon vega/vomma sont trop bruités
    ref = None
    if exercise == "EU" and not has_div:
        ref = BlackScholes(market, option, pricing_date).greeks()
    show_greeks("trinomial",
                price_and_greeks(market, option, pricing_date, nb_step, "tri", prune_threshold=prune,
                                 smoothing=True),
                ref)
    if not has_div:
        show_greeks("binomial", price_and_greeks(market, option, pricing_date, nb_step, "bin", smoothing=True), ref)

    # ---------- Graphiques (optionnels) ----------
    if input("\nGénérer les graphiques (convergence + SVG sur 5 pas) ? (o/n) ").lower() == "o":
        small = Model(pricing_date, 5)
        tri_small = TrinomialTree(market, option, small, prune_threshold=prune)
        tri_small.price()
        TrinomialGraphique(tri_small).to_svg("arbre trinomial.svg")
        TrinomialGraphique(tri_tree).convergence_black_scholes()
        if not has_div:
            bi_small = BinomialTree(market, option, small)
            bi_small.price()
            BinomialGraphique(bi_small).to_svg("arbre binomial.svg")
            BinomialGraphique(BinomialTree(market, option, model)).convergence_black_scholes()
