from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from simulus.core.causal_graph import CausalGraph, CausalNode, Sentiment, NodeType
from simulus.core.montecarlo import SimulationResult


SENTIMENT_COLORS = {
    Sentiment.POSITIVE: "green",
    Sentiment.NEGATIVE: "red",
    Sentiment.NEUTRAL: "yellow",
}

NODE_ICONS = {
    NodeType.ROOT: "(*)",
    NodeType.DECISION: "[?]",
    NodeType.CONSEQUENCE: "-->",
    NodeType.OUTCOME: "[!]",
    NodeType.ACTOR_REACTION: "<@>",
    NodeType.FEEDBACK: "~>~",
}


def _format_probability(prob: float) -> str:
    return f"{prob * 100:.1f}%"


def _build_rich_tree(cg: CausalGraph, node_id: str,
                     parent_tree: Tree | None = None) -> Tree:
    node = cg.get_node(node_id)
    color = SENTIMENT_COLORS.get(node.sentiment, "white")
    icon = NODE_ICONS.get(node.node_type, "   ")

    if node.node_type == NodeType.ROOT:
        label = f"[bold white]{icon} {node.label}[/bold white]"
    else:
        prob_str = _format_probability(node.probability)
        label = f"[{color}]{icon} {node.label}  ({prob_str})[/{color}]"

    if parent_tree is None:
        tree = Tree(label)
    else:
        tree = parent_tree.add(label)

    for child in cg.get_children(node_id):
        _build_rich_tree(cg, child.node_id, tree)

    return tree


def render_tree(cg: CausalGraph, console: Console | None = None,
                animated: bool = True) -> None:
    if console is None:
        console = Console()

    tree = _build_rich_tree(cg, cg.root_id)

    header = Text("SIMULUS -- Decision Tree", style="bold cyan")
    console.print()
    console.print(Panel(header, border_style="cyan", expand=False))
    console.print()

    if animated:
        _render_animated(cg, console)
    else:
        console.print(tree)

    console.print()


def _render_animated(cg: CausalGraph, console: Console) -> None:
    """Render the tree depth by depth with a short delay between levels."""
    max_depth = max(n.depth for n in cg.get_all_nodes())

    for target_depth in range(max_depth + 1):
        partial = _build_partial_tree(cg, cg.root_id, target_depth)
        console.clear()
        header = Text("SIMULUS -- Decision Tree", style="bold cyan")
        console.print()
        console.print(Panel(header, border_style="cyan", expand=False))
        console.print()
        console.print(partial)
        time.sleep(0.4)


def _build_partial_tree(cg: CausalGraph, node_id: str,
                        max_visible_depth: int,
                        parent_tree: Tree | None = None) -> Tree:
    node = cg.get_node(node_id)

    if node.depth > max_visible_depth:
        return parent_tree

    color = SENTIMENT_COLORS.get(node.sentiment, "white")
    icon = NODE_ICONS.get(node.node_type, "   ")

    if node.node_type == NodeType.ROOT:
        label = f"[bold white]{icon} {node.label}[/bold white]"
    else:
        prob_str = _format_probability(node.probability)
        label = f"[{color}]{icon} {node.label}  ({prob_str})[/{color}]"

    if parent_tree is None:
        tree = Tree(label)
    else:
        tree = parent_tree.add(label)

    if node.depth < max_visible_depth:
        for child in cg.get_children(node_id):
            _build_partial_tree(cg, child.node_id, max_visible_depth, tree)

    return tree


def render_monte_carlo_results(result: SimulationResult,
                               console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    console.print(Panel(
        Text("Monte Carlo Simulation Results", style="bold magenta"),
        border_style="magenta",
        expand=False,
    ))
    console.print()
    console.print(f"  Simulations run: [bold]{result.n_simulations:,}[/bold]")
    console.print()

    top = result.top_outcomes(8)
    max_label_len = max(len(label) for label, _ in top) if top else 20
    for label, prob in top:
        bar_len = int(prob * 40)
        bar = "#" * bar_len
        color = "green" if prob > 0.15 else ("yellow" if prob > 0.05 else "red")
        padded_label = label.ljust(max_label_len)
        console.print(f"  {padded_label}  [{color}]{bar}[/{color}]  {prob * 100:.1f}%")

    console.print()

    sentiment_bar = ""
    pos = result.sentiment_distribution.get("positive", 0)
    neg = result.sentiment_distribution.get("negative", 0)
    neu = result.sentiment_distribution.get("neutral", 0)
    sentiment_bar = (
        f"  [green]Positive: {pos * 100:.1f}%[/green]  "
        f"[yellow]Neutral: {neu * 100:.1f}%[/yellow]  "
        f"[red]Negative: {neg * 100:.1f}%[/red]"
    )
    console.print(sentiment_bar)

    score = result.mean_sentiment_score
    score_color = "green" if score > 0.1 else ("red" if score < -0.1 else "yellow")
    console.print(f"  Mean sentiment: [{score_color}]{score:+.3f}[/{score_color}]")
    console.print()
