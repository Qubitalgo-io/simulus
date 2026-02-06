from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.text import Text


def typewriter(text: str, console: Console | None = None,
               delay: float = 0.03, style: str = "") -> None:
    if console is None:
        console = Console()
    for char in text:
        console.print(char, end="", style=style, highlight=False)
        time.sleep(delay)
    console.print()


def fade_in_text(text: str, console: Console | None = None,
                 steps: int = 5) -> None:
    if console is None:
        console = Console()

    shades = ["grey30", "grey50", "grey70", "grey85", "white"]
    actual_steps = min(steps, len(shades))

    for i in range(actual_steps):
        console.print(f"\r[{shades[i]}]{text}[/{shades[i]}]", end="")
        time.sleep(0.08)
    console.print()


def progress_ripple(label: str, width: int = 30,
                    console: Console | None = None) -> None:
    if console is None:
        console = Console()

    import sys
    for i in range(width + 1):
        filled = "#" * i
        empty = "-" * (width - i)
        sys.stdout.write(f"\r  {label} [{filled}{empty}]")
        sys.stdout.flush()
        time.sleep(0.02)
    sys.stdout.write("\n")
    sys.stdout.flush()


def divider(console: Console | None = None, style: str = "dim cyan") -> None:
    if console is None:
        console = Console()
    width = min(console.width, 60)
    console.print(Text("=" * width, style=style))


def header_banner(title: str, subtitle: str = "",
                  console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    divider(console)
    console.print()
    typewriter(f"  {title}", console, delay=0.04, style="bold cyan")
    if subtitle:
        typewriter(f"  {subtitle}", console, delay=0.02, style="dim white")
    console.print()
    divider(console)
    console.print()


def chaos_reveal(perturbation: float, divergence: float,
                 console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    typewriter("  Perturbation applied...", console, delay=0.04, style="bold yellow")
    time.sleep(0.3)

    progress_ripple("Chaos propagating", console=console)
    console.print()

    color = "red" if divergence > 50 else ("yellow" if divergence > 20 else "green")
    console.print(f"  Initial perturbation:  [dim]{perturbation:+.4f}[/dim]")
    console.print(f"  Fate divergence:       [{color}]{divergence:.1f}%[/{color}]")

    if divergence > 70:
        typewriter("  Two completely different lives.", console,
                   delay=0.05, style="bold red")
    elif divergence > 40:
        typewriter("  The futures have meaningfully diverged.", console,
                   delay=0.05, style="bold yellow")
    else:
        typewriter("  The butterfly barely flapped.", console,
                   delay=0.05, style="dim green")
    console.print()
