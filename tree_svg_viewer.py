"""Visualisation SVG des arbres binomiaux et trinomiaux.

Tous les noeuds de la structure sont dessinés. Les noeuds orphelins
(inatteignables depuis la racine) sont affichés en rouge pour repérer
les erreurs de construction de l'arbre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import svgwrite

BINOMIAL_CHILDREN: tuple[str, ...] = ("next_down", "next_up")
TRINOMIAL_CHILDREN: tuple[str, ...] = ("next_down", "next_mid", "next_up")

COLOR_NORMAL_FILL: str = "#DCEEFF"
COLOR_NORMAL_STROKE: str = "#1E3A8A"
COLOR_ORPHAN_FILL: str = "#FFD6D6"
COLOR_ORPHAN_STROKE: str = "#D62728"
COLOR_OK_TEXT: str = "#2CA02C"
EPSILON: float = 1e-12

Position = tuple[float, float]


@dataclass(frozen=True)
class SvgStyle:
    #Paramètres graphiques du dessin.

    x_spacing: float = 90.0
    node_radius: float = 16.0
    node_gap: float = 4.0
    margin: float = 40.0
    font_size: float = 7.0
    branch_width: float = 1.0
    node_stroke: float = 1.5
    max_height: float | None = 1500.0
    log_scale: bool = False


def collect_columns(tree: Any) -> list[list[Any]]:
    """Retourne toutes les colonnes de l'arbre, orphelins compris.

    On remonte les troncs depuis la dernière colonne, puis on parcourt
    chaque colonne du bas vers le haut avec les liens `above`.
    """
    trunks: list[Any] = []
    trunk = tree.last_trunk
    while trunk is not None:
        trunks.append(trunk)
        trunk = trunk.prec_trunk
    trunks.reverse()

    columns: list[list[Any]] = []
    for trunk in trunks:
        column: list[Any] = []
        node = trunk.lowest()
        while node is not None:
            column.append(node)
            node = node.above
        columns.append(column)
    return columns


def find_reached_nodes(root: Any, child_attrs: tuple[str, ...]) -> set[int]:
    #Retourne les identifiants des noeuds atteignables depuis la racine.
    reached: set[int] = {id(root)}
    frontier: list[Any] = [root]
    while frontier:
        next_frontier: list[Any] = []
        for node in frontier:
            for attr in child_attrs:
                child = getattr(node, attr)
                if child is not None and id(child) not in reached:
                    reached.add(id(child))
                    next_frontier.append(child)
        frontier = next_frontier
    return reached


def find_orphans(columns: list[list[Any]], reached: set[int]) -> list[tuple[int, Any]]:
    #Retourne les couples (numéro de colonne, noeud) des orphelins.
    return [
        (index, node)
        for index, column in enumerate(columns)
        for node in column
        if id(node) not in reached
    ]


def vertical_value(node: Any, log_scale: bool) -> float:
    #Ordonnée théorique d'un noeud : son prix ou le log de son prix.
    return math.log(node.St) if log_scale else node.St


def min_vertical_gap(columns: list[list[Any]], log_scale: bool) -> float:
    #Plus petit écart vertical entre deux noeuds d'une même colonne.
    min_gap = float("inf")
    for column in columns:
        values = sorted(vertical_value(node, log_scale) for node in column)
        for low, high in zip(values, values[1:]):
            if high - low > EPSILON:
                min_gap = min(min_gap, high - low)
    return 1.0 if min_gap == float("inf") else min_gap


def compute_layout(
    columns: list[list[Any]], style: SvgStyle
) -> tuple[dict[int, Position], float, float]:
    #Calcule la position de chaque noeud ainsi que la taille du dessin.
    values = [vertical_value(n, style.log_scale) for col in columns for n in col]
    v_min, v_max = min(values), max(values)

    scale = (2 * style.node_radius + style.node_gap) / min_vertical_gap(
        columns, style.log_scale
    )
    if style.max_height is not None and v_max > v_min:
        scale = min(scale, (style.max_height - 2 * style.margin) / (v_max - v_min))

    width = 2 * style.margin + (len(columns) - 1) * style.x_spacing
    height = 2 * style.margin + (v_max - v_min) * scale

    positions: dict[int, Position] = {}
    for index, column in enumerate(columns):
        x = style.margin + index * style.x_spacing
        for node in column:
            y = style.margin + (v_max - vertical_value(node, style.log_scale)) * scale
            positions[id(node)] = (x, y)
    return positions, width, height


def draw_branches(
    dwg: Any,
    columns: list[list[Any]],
    positions: dict[int, Position],
    reached: set[int],
    child_attrs: tuple[str, ...],
    style: SvgStyle,
) -> None:
    #Dessine les branches (rouges en pointillés si le parent est orphelin).
    for column in columns:
        for node in column:
            is_orphan = id(node) not in reached
            drawn: set[int] = set()
            for attr in child_attrs:
                child = getattr(node, attr)
                if child is None or id(child) in drawn or id(child) not in positions:
                    continue
                drawn.add(id(child))
                options: dict[str, Any] = {
                    "stroke": COLOR_ORPHAN_STROKE if is_orphan else "black",
                    "stroke_width": style.branch_width,
                }
                if is_orphan:
                    options["stroke_dasharray"] = "4,3"
                dwg.add(
                    dwg.line(
                        start=positions[id(node)], end=positions[id(child)], **options
                    )
                )


def draw_node(dwg: Any, node: Any, position: Position, is_orphan: bool,
              bold: bool, style: SvgStyle) -> None:
    #Dessine un noeud : cercle, prix du sous-jacent et valeur de l'option.
    x, y = position
    dwg.add(
        dwg.circle(
            center=position,
            r=style.node_radius,
            fill=COLOR_ORPHAN_FILL if is_orphan else COLOR_NORMAL_FILL,
            stroke=COLOR_ORPHAN_STROKE if is_orphan else COLOR_NORMAL_STROKE,
            stroke_width=style.node_stroke,
        )
    )
    weight = "bold" if bold else "normal"
    lines = ((f"S={node.St:.2f}", -0.3), (f"V={node.value:.2f}", 1.0))
    for text, offset in lines:
        dwg.add(
            dwg.text(
                text,
                insert=(x, y + offset * style.font_size),
                text_anchor="middle",
                font_size=f"{style.font_size}px",
                font_weight=weight,
            )
        )


def draw_nodes(dwg: Any,columns: list[list[Any]],positions: dict[int, Position],
               reached: set[int],col_div: int | None,style: SvgStyle) -> None:
    #Dessine tous les noeuds (colonne du dividende en gras).
    for index, column in enumerate(columns):
        bold = col_div is not None and index == col_div
        for node in column:
            is_orphan = id(node) not in reached
            draw_node(dwg, node, positions[id(node)], is_orphan, bold, style)


def describe_orphans(orphans: list[tuple[int, Any]]) -> str:
    #Résumé textuel des orphelins pour l'affichage console.
    if not orphans:
        return "aucun orphelin"
    details = ", ".join(
        f"col {index}: S={node.St:.2f} (proba={node.proba:.2e})"
        for index, node in orphans
    )
    return f"{len(orphans)} orphelin(s) -> {details}"


def draw_tree_svg(
    tree: Any,
    filename: str,
    child_attrs: tuple[str, ...],
    style: SvgStyle = SvgStyle(),
) -> str:
    """Génère le SVG d'un arbre déjà calculé.

    Args:
        tree: arbre sur lequel `price()` a été appelé.
        filename: chemin du fichier SVG à écrire.
        child_attrs: attributs des fils (binomial ou trinomial).
        style: paramètres graphiques.
    """
    if tree.last_trunk is None:
        raise ValueError("L'arbre n'a pas encore été calculé : appeler price().")

    columns = collect_columns(tree)
    reached = find_reached_nodes(tree.root, child_attrs)
    orphans = find_orphans(columns, reached)
    positions, width, height = compute_layout(columns, style)

    dwg = svgwrite.Drawing(filename, size=(f"{width}px", f"{height}px"))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
    draw_branches(dwg, columns, positions, reached, child_attrs, style)
    draw_nodes(dwg, columns, positions, reached, getattr(tree, "col_div", None), style)
    dwg.add(
        dwg.text(
            f"{len(orphans)} noeud(s) orphelin(s) en rouge",
            insert=(style.margin / 2, style.font_size * 2),
            font_size=f"{style.font_size * 1.25}px",
            fill=COLOR_ORPHAN_STROKE if orphans else COLOR_OK_TEXT,
        )
    )
    dwg.save()
    return f"Arbre sauvegardé dans {filename} : {describe_orphans(orphans)}"


def binomial_tree_to_svg(
    tree: Any, filename: str = "binomial_tree.svg", style: SvgStyle = SvgStyle()
) -> str:
    #Dessine un arbre binomial déjà calculé.
    return draw_tree_svg(tree, filename, BINOMIAL_CHILDREN, style)


def trinomial_tree_to_svg(
    tree: Any, filename: str = "trinomial_tree.svg", style: SvgStyle = SvgStyle()
) -> str:
    #Dessine un arbre trinomial déjà calculé.
    return draw_tree_svg(tree, filename, TRINOMIAL_CHILDREN, style)
