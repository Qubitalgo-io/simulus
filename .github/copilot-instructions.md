# Copilot Instructions for Simulus

## Code Style

- No emojis anywhere in the codebase. Not in comments, not in strings, not in docs.
- Use minimal comments. Code should be self-documenting through clear naming.
- Only add a comment when the logic is genuinely non-obvious.
- Prefer type hints over docstrings for function signatures.
- Use lowercase_snake_case for functions and variables.
- Use PascalCase for classes.
- Use UPPER_SNAKE_CASE for constants.

## Architecture

Simulus is a seed-reproducible scenario simulation engine that models the sensitive
dependence of human decisions on initial conditions. It is a pure Python project.

### Core Engine (`simulus/core/`)
- `parser.py` -- NLP situation parser using spaCy. Extracts actors, stakes, environment, emotional state.
- `causal_graph.py` -- Builds a directed acyclic graph of decisions and consequences using networkx.
- `bayesian.py` -- Parameterized probability scoring engine. Computes conditional probabilities for each branch using domain priors, Markov transitions, and heuristic modifiers. Not Bayesian in the textbook sense.
- `chaos.py` -- Sensitive dependence module. Applies Lyapunov-style chaos multipliers at each depth level.
- `montecarlo.py` -- Monte Carlo simulator. Runs N simulations to produce statistical distributions.
- `seed.py` -- Seed-reproducible random number manager. Same seed and input must always produce the same output.

### Renderer (`simulus/renderer/`)
- `tree.py` -- Animated ASCII tree rendering using rich.
- `effects.py` -- Visual effects: typewriter text, color gradients, particle ripples.
- `rewind.py` -- Split-screen replay mode showing two divergent scenario branches side by side.

### CLI (`simulus/cli.py`)
- Entry point. Uses click for argument parsing.
- Interactive mode: user describes a situation, engine projects scenario branches.
- Replay mode: re-run with same seed, or rewind to a branch point and take the other path.
- Butterfly mode: apply a small perturbation and compare trajectory divergence.

## Design Principles

- Determinism is sacred. Given the same seed and input, the output must be identical.
- The probability model must be mathematically grounded, not heuristic guesswork.
- The simulation depth is 6 levels.
- The terminal UI should feel cinematic but never sacrifice clarity for flash.
- Keep dependencies minimal: spaCy, rich, numpy, networkx, click.
- All randomness flows through the seed manager. Never call random.random() directly.

## Testing

- Tests go in `tests/`.
- Test determinism explicitly: same seed must produce byte-identical output.
- Test butterfly divergence: a small perturbation must produce measurably different outcomes.

## Project Context

Simulus is a philosophical and computational demonstration that the universe may be
deterministic, but prediction requires omniscience. Small changes in initial conditions
(the butterfly effect) lead to radically different futures. The project is intended to
be open sourced.
