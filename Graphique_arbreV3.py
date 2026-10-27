import svgwrite
import matplotlib.pyplot as plt
import numpy as np
import math

from Bin_Tri_treeV4 import (
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
    
    
