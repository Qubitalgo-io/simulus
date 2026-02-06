from __future__ import annotations

import click
from rich.console import Console

from simulus.seed import SeedManager
from simulus.core.parser import parse_situation
from simulus.core.causal_graph import build_causal_graph
from simulus.core.bayesian import update_graph_probabilities, expected_sentiment_score
from simulus.core.chaos import create_perturbed_graph, fate_divergence_score
from simulus.core.montecarlo import run_monte_carlo
from simulus.renderer.tree import render_tree, render_monte_carlo_results
from simulus.renderer.effects import header_banner, chaos_reveal, typewriter
from simulus.renderer.rewind import render_rewind


SIMULATION_DEPTH = 6
DEFAULT_PERTURBATION = 0.01
MONTE_CARLO_RUNS = 10_000


@click.command()
@click.option("--scenario", "-s", type=str, default=None,
              help="Describe a situation to simulate.")
@click.option("--seed", type=int, default=None,
              help="Deterministic seed. Same seed + input = same future.")
@click.option("--butterfly", "-b", is_flag=True, default=False,
              help="Enable butterfly mode: apply a small perturbation and compare.")
@click.option("--perturbation", "-p", type=float, default=DEFAULT_PERTURBATION,
              help="Size of butterfly perturbation.")
@click.option("--rewind", "-r", type=int, default=None,
              help="Rewind to decision point N and take the other path.")
@click.option("--depth", "-d", type=int, default=SIMULATION_DEPTH,
              help="Simulation depth (default 6).")
@click.option("--monte-carlo", "-m", type=int, default=MONTE_CARLO_RUNS,
              help="Number of Monte Carlo simulations.")
@click.option("--no-animate", is_flag=True, default=False,
              help="Disable animation.")
def main(scenario: str | None, seed: int | None, butterfly: bool,
         perturbation: float, rewind: int | None, depth: int,
         monte_carlo: int, no_animate: bool) -> None:

    console = Console()
    seed_mgr = SeedManager(seed=seed)

    header_banner(
        "S I M U L U S",
        "The Butterfly Effect Engine",
        console,
    )

    if scenario is None:
        typewriter("Describe a situation:", console, delay=0.03, style="bold white")
        console.print()
        scenario = console.input("  [bold cyan]> [/bold cyan]")

    console.print()
    typewriter(f'  Parsing: "{scenario}"', console, delay=0.02, style="dim")
    context = parse_situation(scenario)
    console.print(f"  [dim]Domain: {context.domain} | Mood: {context.emotional_state}[/dim]")
    console.print(f"  [dim]Stakes: {', '.join(context.stakes)}[/dim]")

    if context.actor_profiles:
        actor_names = [a.name for a in context.actor_profiles]
        console.print(f"  [dim]Actors: {', '.join(actor_names)}[/dim]")

    if context.conflict_vectors:
        for cv in context.conflict_vectors[:2]:
            console.print(f"  [dim]Conflict: {cv.actor_a} vs {cv.actor_b} ({cv.nature})[/dim]")

    console.print(f"  [dim]Volatility: {context.compound_volatility:.2f} | Severity: {context.stake_severity:.2f}[/dim]")
    console.print(f"  [dim]Seed: {seed_mgr.base_seed}[/dim]")
    console.print()

    typewriter("  Building causal graph...", console, delay=0.02, style="dim")
    graph = build_causal_graph(context, seed_mgr, max_depth=depth)

    bayesian_seed = seed_mgr.fork("bayesian")
    update_graph_probabilities(graph, context.domain, context.emotional_state,
                               bayesian_seed, context=context)

    render_tree(graph, console, animated=not no_animate)

    typewriter(f"  Running {monte_carlo:,} Monte Carlo simulations...",
               console, delay=0.02, style="dim")
    mc_seed = seed_mgr.fork("montecarlo")
    mc_result = run_monte_carlo(graph, mc_seed, n_simulations=monte_carlo)
    render_monte_carlo_results(mc_result, console)

    sentiment = expected_sentiment_score(graph)
    sentiment_color = "green" if sentiment > 0.1 else ("red" if sentiment < -0.1 else "yellow")
    console.print(f"  Overall fate score: [{sentiment_color}]{sentiment:+.3f}[/{sentiment_color}]")
    console.print()

    if butterfly:
        _run_butterfly_mode(graph, perturbation, seed_mgr, console, context)

    if rewind is not None:
        _run_rewind_mode(graph, context, seed_mgr, rewind, depth, console)

    typewriter("  The future is determined. The seed is set.", console,
               delay=0.05, style="dim white")
    console.print()


def _run_butterfly_mode(graph, perturbation, seed_mgr, console, context=None):
    perturbed_seed = seed_mgr.fork("butterfly")
    perturbed = create_perturbed_graph(graph, perturbation, perturbed_seed,
                                       context=context)
    divergence = fate_divergence_score(graph, perturbed)
    chaos_reveal(perturbation, divergence, console)

    render_rewind(
        graph, perturbed, divergence,
        label_a="Original Reality",
        label_b="Perturbed Reality",
        console=console,
    )


def _run_rewind_mode(graph, context, seed_mgr, rewind_point, depth, console):
    # rebuild the graph with a different seed fork to simulate the alternate path
    alt_seed = seed_mgr.fork(f"rewind_{rewind_point}")
    alt_graph = build_causal_graph(context, alt_seed, max_depth=depth)

    alt_bayesian_seed = alt_seed.fork("bayesian_alt")
    update_graph_probabilities(alt_graph, context.domain,
                               context.emotional_state, alt_bayesian_seed,
                               context=context)

    divergence = fate_divergence_score(graph, alt_graph)

    render_rewind(
        graph, alt_graph, divergence,
        label_a="Path Taken",
        label_b="Path Not Taken",
        console=console,
    )


if __name__ == "__main__":
    main()
