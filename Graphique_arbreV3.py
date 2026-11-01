import svgwrite
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from Bin_Tri_treeV5 import (
    Market,
    Option,
    Model,
    BiNode,
    TriNode,
    TrunkBiNode,
    TrunkTriNode,
    BinomialTree,
    TrinomialTree,
    BlackScholes,
    price_binomial_tree,
    price_trinomial_tree
)


# ---------- Temps de calcul avec / sans pruning ----------

def _mesurer_temps(price_func, market, option, pricing_date, nb_pas_list,
                   prune_threshold, smoothing=False, repeats=3):
    """
    Mesure le temps de calcul (en secondes) de price_func pour chaque N de nb_pas_list,
    sans pruning puis avec pruning. On garde le meilleur de `repeats` essais pour limiter
    le bruit (garbage collector, autres processus...). Un calcul à blanc est fait avant
    pour ne pas compter le premier appel (imports, caches).
    """
    def chrono(n, prune):
        meilleur = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            price_func(market, option, pricing_date, n, prune, smoothing)
            meilleur = min(meilleur, time.perf_counter() - t0)
        return meilleur

    price_func(market, option, pricing_date, 10, None, smoothing)   # échauffement

    sans = [chrono(n, None) for n in nb_pas_list]
    avec = [chrono(n, prune_threshold) for n in nb_pas_list]
    return sans, avec


def _tracer_temps(ax, nb_pas_list, sans, avec, prune_threshold, titre):
    """Trace les deux courbes (sans / avec pruning) en échelle log-log sur l'axe ax."""
    ax.plot(nb_pas_list, sans, marker="o", label="sans pruning")
    ax.plot(nb_pas_list, avec, marker="o", label=f"pruning {prune_threshold:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nombre de pas N")
    ax.set_ylabel("Temps de calcul (s)")
    ax.set_title(titre)
    ax.grid(True, which="both")
    ax.legend()


def temps_calcul_bin_tri(market, option, pricing_date,
                         nb_pas_list=(50, 100, 200, 400, 700, 1000),
                         prune_threshold=1e-7, smoothing=False, repeats=3,
                         filename="temps_calcul_bin_tri.png"):
    """
    Une seule figure, deux graphiques côte à côte (binomial | trinomial) :
    temps de calcul avec et sans pruning en fonction du nombre de pas N.
    Le binomial ignore le dividende discret : à utiliser sans dividende pour comparer.
    """
    bin_sans, bin_avec = _mesurer_temps(price_binomial_tree, market, option, pricing_date,
                                        nb_pas_list, prune_threshold, smoothing, repeats)
    tri_sans, tri_avec = _mesurer_temps(price_trinomial_tree, market, option, pricing_date,
                                        nb_pas_list, prune_threshold, smoothing, repeats)

    fig, (ax_bin, ax_tri) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    _tracer_temps(ax_bin, nb_pas_list, bin_sans, bin_avec, prune_threshold,
                  "Temps de calcul de l'arbre binomial")
    _tracer_temps(ax_tri, nb_pas_list, tri_sans, tri_avec, prune_threshold,
                  "Temps de calcul de l'arbre trinomial")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return f"figure sauvée dans '{filename}'"


def etude_ordre_convergence(market, option, pricing_date,
                            nb_pas_list=tuple(range(10, 301, 10)),
                            prune_threshold=None, smoothing=True,
                            filename="ordre_convergence_bin_tri.png"):
    """
    Étude de la vitesse de convergence vers Black-Scholes, binomial et trinomial
    (une ligne par arbre, échelle propre à chaque graphique). On trace écart * sqrt(N), écart * N et écart * N^2,
    où écart = prix arbre - prix Black-Scholes.

    Lecture : si écart ~ C / N^p, alors écart * N^q est
      - décroissant vers 0 si q < p,
      - constant (= C)      si q = p   -> c'est l'ordre de convergence,
      - croissant           si q > p (linéairement pour q = p + 1).

    smoothing=True (défaut) : dernière étape valorisée par Black-Scholes, ce qui supprime
    les oscillations dues à la position du strike par rapport aux nœuds.
    Black-Scholes est la référence : option européenne sans dividende uniquement.
    """
    if option.exercise != "EU" or market.q_ex_div_date is not None:
        raise ValueError("Black-Scholes n'est la référence que pour un européen sans dividende.")

    price_bs = BlackScholes(market, option, pricing_date).price_option()
    N = np.array(nb_pas_list, dtype=float)

    gaps = {
        "Binomial": np.array([price_binomial_tree(market, option, pricing_date, int(n), None, smoothing) - price_bs
                              for n in nb_pas_list]),
        "Trinomial": np.array([price_trinomial_tree(market, option, pricing_date, int(n), prune_threshold, smoothing) - price_bs
                               for n in nb_pas_list]),
    }
    couleurs = {"Binomial": "tab:blue", "Trinomial": "tab:orange"}

    facteurs = [
        (np.sqrt(N), r"$\sqrt{N}\,(V_N - V_{BS})$"),
        (N, r"$N\,(V_N - V_{BS})$"),
        (N ** 2, r"$N^2\,(V_N - V_{BS})$"),
    ]

    # 2 lignes (binomial / trinomial) x 3 colonnes (sqrt(N), N, N^2), échelle propre à chaque graphique
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for ligne, (name, gap) in enumerate(gaps.items()):
        for col, (facteur, titre) in enumerate(facteurs):
            ax = axes[ligne, col]
            ax.plot(N, facteur * gap, color=couleurs[name])
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_title(f"{name} : {titre}")
            ax.set_xlabel("Nombre de pas N")
            ax.set_ylabel(titre)
            ax.grid(True)

    fig.suptitle("Ordre de convergence vers Black-Scholes : arbre binomial et trinomial")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return f"figure sauvée dans '{filename}'"


def erreur_vs_strike(market, option, pricing_date, nb_pas=100, strikes=None,
                     prune_threshold=None, smoothing=False,
                     filename="erreur_vs_strike_bin_tri.png"):
    """
    Erreur de l'arbre (prix arbre - prix Black-Scholes) en fonction du strike K,
    pour un nombre de pas nb_pas fixé, binomial et trinomial.

    Si les deux erreurs ont le même ordre de grandeur, elles sont tracées sur un seul
    graphique ; si l'une écrase l'autre (rapport des erreurs max > 10), on trace deux
    graphiques côte à côte, chacun avec sa propre échelle.
    Par défaut strikes = 70 % à 130 % de S0. Option européenne sans dividende uniquement.
    """
    if option.exercise != "EU" or market.q_ex_div_date is not None:
        raise ValueError("Black-Scholes n'est la référence que pour un européen sans dividende.")

    if strikes is None:
        strikes = np.linspace(0.7 * market.s0, 1.3 * market.s0, 121)
    strikes = np.array(strikes, dtype=float)

    err_bin, err_tri = [], []
    for K in strikes:
        opt = Option(strike=float(K), t=option.t, CP=option.CP, exercise=option.exercise,
                     digital=option.digital, payout=option.payout)
        bs = BlackScholes(market, opt, pricing_date).price_option()
        err_bin.append(price_binomial_tree(market, opt, pricing_date, nb_pas, None, smoothing) - bs)
        err_tri.append(price_trinomial_tree(market, opt, pricing_date, nb_pas, prune_threshold, smoothing) - bs)
    err_bin, err_tri = np.array(err_bin), np.array(err_tri)

    m_bin, m_tri = np.max(np.abs(err_bin)), np.max(np.abs(err_tri))
    meme_echelle = max(m_bin, m_tri) <= 10 * max(min(m_bin, m_tri), 1e-300)

    titre = f"Erreur de l'arbre en fonction du strike (N = {nb_pas})"
    ylabel = r"$V_N - V_{BS}$"

    if meme_echelle:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(strikes, err_bin, label="Binomial", color="tab:blue")
        ax.plot(strikes, err_tri, label="Trinomial", color="tab:orange")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(market.s0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Strike K")
        ax.set_ylabel(ylabel)
        ax.set_title(titre)
        ax.grid(True)
        ax.legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for ax, name, err, couleur in zip(axes, ("Binomial", "Trinomial"),
                                          (err_bin, err_tri), ("tab:blue", "tab:orange")):
            ax.plot(strikes, err, color=couleur)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axvline(market.s0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Strike K")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}")
            ax.grid(True)
        fig.suptitle(titre)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return f"figure sauvée dans '{filename}'" + ("" if meme_echelle else " (deux graphiques : échelles trop différentes)")


#classe pour les différents graphiques
class BinomialGraphique():
    
    def __init__(self, market, option, pricing_date, prune_threshold=None, smoothing=False):
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.prune_threshold = prune_threshold
        self.smoothing = smoothing

    def convergence_black_scholes(self, max_nb_pas):
        """
        Affiche la convergence du prix de l'arbre binomial
        vers le prix Black-Scholes.
        """

        market = self.market
        option = self.option
        pricing_date = self.pricing_date
        prune_threshold = self.prune_threshold
        smoothing = self.smoothing

        # Prix Black-Scholes de référence 
        bs = BlackScholes(
            market,
            option,
            pricing_date
        )
        
        price_bs = bs.price_option()

        # Nombre de pas
        N_values = np.arange(10, max_nb_pas)
        errors = []

        # Calcul de sqrt(N) * |V_binomial - V_BS|
        for n in N_values:

            price = price_binomial_tree(
                market,
                option,
                pricing_date,
                n,
                prune_threshold,
                smoothing
            )

            error = n * (price - price_bs)
            errors.append(error)

        # Graphique
        plt.figure(figsize=(10, 6))

        plt.scatter(
            N_values,
            errors,
            label=r"$N\,(V_N - V_{BS})$"
        )

        plt.xlabel("Nombre de pas N")
        plt.ylabel(r"$N\,(V_N - V_{BS})$")
        plt.title("Convergence de l'arbre binomial vers Black-Scholes")
        plt.grid(True)
        plt.legend()

        plt.savefig("CV_BS_bin.png")
        return("figure sauvée dans 'CV_BS_bin.png'")

    def temps_calcul(self, nb_pas_list=(50, 100, 200, 400, 700, 1000), prune_threshold=1e-7,
                     repeats=3, filename="temps_calcul_bin.png"):
        """
        Temps de calcul de l'arbre binomial avec et sans pruning, en fonction du nombre de pas.
        prune_threshold : seuil utilisé pour la courbe "avec pruning".
        repeats : nombre d'essais par point (on garde le plus rapide).
        """
        sans, avec = _mesurer_temps(price_binomial_tree, self.market, self.option, self.pricing_date,
                                    nb_pas_list, prune_threshold, self.smoothing, repeats)

        fig, ax = plt.subplots(figsize=(10, 6))
        _tracer_temps(ax, nb_pas_list, sans, avec, prune_threshold,
                      "Temps de calcul de l'arbre binomial, avec et sans pruning")
        fig.savefig(filename)
        plt.close(fig)
        return f"figure sauvée dans '{filename}'"

    def to_svg(self,nb_pas, filename="binomial_tree.svg"):
        """
        Créer une représentation SVG de l'arbre binomial.
        """
        model = Model(self.pricing_date,nb_pas)
        tree = BinomialTree(self.market, self.option, model, prune_threshold=self.prune_threshold, smoothing=self.smoothing)
        price = tree.price()

        # L'arbre doit avoir été calculé avant
        if tree.last_trunk is None:
            raise ValueError(
                "L'arbre n'a pas encore été calculé. "
                "Utilise tree.price() avant."
            )

        # Paramètres graphiques
        x_spacing = 180
        y_spacing = 100

        node_radius = 35

        margin_x = 80
        margin_y = 80

        # Récupération des colonnes
        columns = [[tree.root]]

        for i in range(tree.model.nb_step):

            current_column = columns[-1]
            next_column = []

            for node in current_column:

                if node.next_down is not None:
                    if node.next_down not in next_column:
                        next_column.append(node.next_down)

                if node.next_up is not None:
                    if node.next_up not in next_column:
                        next_column.append(node.next_up)

            columns.append(next_column)

        # Position des nœuds
        positions = {}

        max_nodes = max(
            len(column)
            for column in columns
        )

        width = (
            2 * margin_x
            + tree.model.nb_step * x_spacing
        )

        height = (
            2 * margin_y
            + (max_nodes - 1) * y_spacing
        )

        for i, column in enumerate(columns):

            column_height = (
                (len(column) - 1)
                * y_spacing
            )

            y_start = (
                height / 2
                - column_height / 2
            )

            for j, node in enumerate(column):

                x = margin_x + i * x_spacing

                y = (
                    y_start
                    + (len(column) - 1 - j)
                    * y_spacing
                )

                positions[id(node)] = (x, y)

        # Création du SVG
        dwg = svgwrite.Drawing(
            filename,
            size=(f"{width}px", f"{height}px")
        )

        # Fond
        dwg.add(
            dwg.rect(
                insert=(0, 0),
                size=(width, height),
                fill="white"
            )
        )

        # Branches
        for column in columns:

            for node in column:

                x1, y1 = positions[id(node)]

                if node.next_up is not None:

                    x2, y2 = positions[id(node.next_up)]

                    dwg.add(
                        dwg.line(
                            start=(x1, y1),
                            end=(x2, y2),
                            stroke="black",
                            stroke_width=2
                        )
                    )

                if node.next_down is not None:

                    x2, y2 = positions[id(node.next_down)]

                    dwg.add(
                        dwg.line(
                            start=(x1, y1),
                            end=(x2, y2),
                            stroke="black",
                            stroke_width=2
                        )
                    )

        # Nœuds
        # Colonne impactée par le dividende (None s'il n'y a pas de dividende)
        col_div = getattr(tree, "col_div", None)

        # Nœuds
        for i, column in enumerate(columns):

            weight = "bold" if (col_div is not None and i == col_div) else "normal"

            for node in column:

                x, y = positions[id(node)]

                # Cercle (inchangé)
                dwg.add(
                    dwg.circle(
                        center=(x, y),
                        r=node_radius,
                        fill="#DCEEFF",
                        stroke="#1E3A8A",
                        stroke_width=2
                    )
                )

                # Prix du sous-jacent
                dwg.add(
                    dwg.text(
                        f"S={node.St:.2f}",
                        insert=(x, y - 5),
                        text_anchor="middle",
                        font_size="12px",
                        font_weight=weight
                    )
                )

                # Prix de l'option
                dwg.add(
                    dwg.text(
                        f"V={node.value:.2f}",
                        insert=(x, y + 13),
                        text_anchor="middle",
                        font_size="12px",
                        font_weight=weight
                    )
                )

        # Sauvegarde
        dwg.save()

        return(f"Arbre binomial sauvegardé dans : {filename}")
        
class TrinomialGraphique():
    
    def __init__(self, market, option, pricing_date, prune_threshold=None, smoothing=False):
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.prune_threshold = prune_threshold
        self.smoothing = smoothing

    def convergence_black_scholes(self,max_nb_pas):
        """
        Affiche la convergence du prix de l'arbre trinomial
        vers le prix Black-Scholes.
        """

        market = self.market
        option = self.option
        pricing_date = self.pricing_date
        prune_threshold = self.prune_threshold
        smoothing = self.smoothing 

        # Prix Black-Scholes de référence
        bs = BlackScholes(market, option, pricing_date)
        price_bs = bs.price_option()

        # Nombre de pas
        N_values = np.arange(10, max_nb_pas)
        errors = []

        # Calcul de N * |V_trinomial - V_BS| (erreur en 1/N)
        for n in N_values:
            price = price_trinomial_tree(market, option, pricing_date, n, prune_threshold, smoothing)
            error = n * (price - price_bs)
            errors.append(error)

        # Graphique
        plt.figure(figsize=(10, 6))

        plt.scatter(
            N_values,
            errors,
            label=r"$N\,(V_N - V_{BS})$"
        )

        plt.xlabel("Nombre de pas N")
        plt.ylabel(r"$N\,(V_N - V_{BS})$")
        plt.title("Convergence de l'arbre trinomial vers Black-Scholes")
        plt.grid(True)
        plt.legend()

        plt.savefig("CV_BS_trin.png")
        return("figure sauvée dans 'CV_BS_trin.png'")

    def temps_calcul(self, nb_pas_list=(50, 100, 200, 400, 700, 1000), prune_threshold=1e-7,
                     repeats=3, filename="temps_calcul_trin.png"):
        """
        Temps de calcul de l'arbre trinomial avec et sans pruning, en fonction du nombre de pas.
        prune_threshold : seuil utilisé pour la courbe "avec pruning".
        repeats : nombre d'essais par point (on garde le plus rapide).
        """
        sans, avec = _mesurer_temps(price_trinomial_tree, self.market, self.option, self.pricing_date,
                                    nb_pas_list, prune_threshold, self.smoothing, repeats)

        fig, ax = plt.subplots(figsize=(10, 6))
        _tracer_temps(ax, nb_pas_list, sans, avec, prune_threshold,
                      "Temps de calcul de l'arbre trinomial, avec et sans pruning")
        fig.savefig(filename)
        plt.close(fig)
        return f"figure sauvée dans '{filename}'"

    def to_svg(self, nb_pas, filename="trinomial_tree.svg"):
        """
        Génère une représentation SVG de l'arbre trinomial.
        """

        model = Model(self.pricing_date,nb_pas)
        tree = TrinomialTree(self.market, self.option, model, prune_threshold=self.prune_threshold, smoothing=self.smoothing)
        price = tree.price()
        
        # L'arbre doit avoir été calculé avant
        if tree.last_trunk is None:
            raise ValueError(
                "L'arbre n'a pas encore été calculé. "
                "Utilise tree.price() avant."
            )

        # Paramètres graphiques
        x_spacing = 180
        y_spacing = 100

        node_radius = 35

        margin_x = 80
        margin_y = 80

        # Récupération des colonnes (chacune ordonnée du bas vers le haut)
        columns = [[tree.root]]

        for i in range(tree.model.nb_step):

            current_column = columns[-1]

            # nœud le plus bas de la colonne suivante
            node = current_column[0].next_down
            next_column = []

            # on remonte la colonne suivante avec les liens "above"
            while node is not None:
                next_column.append(node)
                node = node.above

            columns.append(next_column)

        # Position des nœuds
        positions = {}

        max_nodes = max(len(column) for column in columns)

        width = 2 * margin_x + tree.model.nb_step * x_spacing
        height = 2 * margin_y + (max_nodes - 1) * y_spacing

        for i, column in enumerate(columns):

            column_height = (len(column) - 1) * y_spacing
            y_start = height / 2 - column_height / 2

            for j, node in enumerate(column):

                x = margin_x + i * x_spacing
                y = y_start + (len(column) - 1 - j) * y_spacing

                positions[id(node)] = (x, y)

        # Création du SVG
        dwg = svgwrite.Drawing(
            filename,
            size=(f"{width}px", f"{height}px")
        )

        # Fond
        dwg.add(
            dwg.rect(insert=(0, 0), size=(width, height), fill="white")
        )

        # Branches (up, mid, down)
        for column in columns:

            for node in column:

                x1, y1 = positions[id(node)]

                for child in (node.next_up, node.next_mid, node.next_down):

                    if child is not None:

                        x2, y2 = positions[id(child)]

                        dwg.add(
                            dwg.line(
                                start=(x1, y1),
                                end=(x2, y2),
                                stroke="black",
                                stroke_width=2
                            )
                        )

        # Nœuds
        for column in columns:

            for node in column:

                x, y = positions[id(node)]

                # Cercle
                dwg.add(
                    dwg.circle(
                        center=(x, y),
                        r=node_radius,
                        fill="#DCEEFF",
                        stroke="#1E3A8A",
                        stroke_width=2
                    )
                )

                # Prix du sous-jacent
                dwg.add(
                    dwg.text(
                        f"S={node.St:.2f}",
                        insert=(x, y - 5),
                        text_anchor="middle",
                        font_size="12px"
                    )
                )

                # Prix de l'option
                dwg.add(
                    dwg.text(
                        f"V={node.value:.2f}",
                        insert=(x, y + 13),
                        text_anchor="middle",
                        font_size="12px"
                    )
                )

        # Sauvegarde
        dwg.save()

        return(f"Arbre trinomial sauvegardé dans : {filename}")
    
    
