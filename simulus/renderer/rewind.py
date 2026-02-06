from __future__ import annotations

import time

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from simulus.core.causal_graph import CausalGraph, Sentiment, NodeType
from simulus.renderer.effects import typewriter, divider


SENTIMENT_COLORS = {
    Sentiment.POSITIVE: "green",
    Sentiment.NEGATIVE: "red",
    Sentiment.NEUTRAL: "yellow",
}


def _build_compact_tree(cg: CausalGraph, node_id: str,
                        parent_tree: Tree | None = None,
                        max_depth: int | None = None) -> Tree:
    node = cg.get_node(node_id)
    if max_depth is not None and node.depth > max_depth:
        return parent_tree

    color = SENTIMENT_COLORS.get(node.sentiment, "white")

    if node.node_type == NodeType.ROOT:
        label = f"[bold white]{node.label}[/bold white]"
    else:
        prob_pct = f"{node.probability * 100:.1f}%"
        label = f"[{color}]{node.label} ({prob_pct})[/{color}]"

    if parent_tree is None:
        tree = Tree(label)
    else:
        tree = parent_tree.add(label)

    for child in cg.get_children(node_id):
        _build_compact_tree(cg, child.node_id, tree, max_depth)

    return tree


def render_rewind(graph_a: CausalGraph, graph_b: CausalGraph,
                  divergence_score: float,
                  label_a: str = "Reality A",
                  label_b: str = "Reality B",
                  console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    divider(console, style="bold magenta")
    typewriter("  REWIND MODE -- Two realities diverge", console,
               delay=0.04, style="bold magenta")
    divider(console, style="bold magenta")
    console.print()

    tree_a = _build_compact_tree(graph_a, graph_a.root_id, max_depth=4)
    tree_b = _build_compact_tree(graph_b, graph_b.root_id, max_depth=4)

    panel_a = Panel(tree_a, title=f"[bold cyan]{label_a}[/bold cyan]",
                    border_style="cyan", expand=True, width=45)
    panel_b = Panel(tree_b, title=f"[bold yellow]{label_b}[/bold yellow]",
                    border_style="yellow", expand=True, width=45)

    console.print(Columns([panel_a, panel_b], padding=2))
    console.print()

    color = "red" if divergence_score > 50 else (
        "yellow" if divergence_score > 20 else "green")
    console.print(
        f"  Fate Divergence: [{color}][bold]{divergence_score:.1f}%[/bold][/{color}]"
    )

    console.print()
    _render_timeline_comparison(graph_a, graph_b, console)
    console.print()


def _render_timeline_comparison(graph_a: CausalGraph, graph_b: CausalGraph,
                                console: Console) -> None:
    """Show a depth-by-depth comparison of the most probable path in each reality."""
    console.print("  [dim]Timeline comparison (most probable path):[/dim]")
    console.print()

    path_a = _most_probable_path(graph_a)
    path_b = _most_probable_path(graph_b)

    max_len = max(len(path_a), len(path_b))

    for i in range(max_len):
        node_a = path_a[i] if i < len(path_a) else None
        node_b = path_b[i] if i < len(path_b) else None

        left = ""
        right = ""

        if node_a:
            ca = SENTIMENT_COLORS.get(node_a.sentiment, "white")
            left = f"[{ca}]{node_a.label}[/{ca}]"
        else:
            left = "[dim]---[/dim]"

        if node_b:
            cb = SENTIMENT_COLORS.get(node_b.sentiment, "white")
            right = f"[{cb}]{node_b.label}[/{cb}]"
        else:
            right = "[dim]---[/dim]"

        diverged = ""
        if node_a and node_b and node_a.label != node_b.label:
            diverged = " [bold red]<< DIVERGED[/bold red]"

        console.print(f"    Depth {i}: {left:40s}  |  {right}{diverged}")
        time.sleep(0.15)


def _most_probable_path(cg: CausalGraph) -> list:
    path = []
    current_id = cg.root_id
    while True:
        node = cg.get_node(current_id)
        path.append(node)
        children = cg.get_children(current_id)
        if not children:
            break
        best_child = max(children, key=lambda c: c.probability)
        current_id = best_child.node_id
    return path
