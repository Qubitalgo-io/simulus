# Simulus

**The Sensitive Dependence Engine**

A seed-reproducible scenario simulation engine that models how small decisions
cascade into divergent outcome trajectories. Describe any situation, make a choice,
and watch the consequences branch.

---

## What is this?

Simulus is a terminal-based simulation tool built on a philosophical premise:
decisions have consequences that cascade through time, and those consequences
exhibit sensitive dependence on initial conditions -- small changes at the start
produce large differences downstream.

You describe a situation in plain English. Simulus parses it, builds a causal graph
of possible decisions and their consequences, assigns probabilities using a
parameterized scoring model, and projects scenario branches 6 levels deep.

Then it demonstrates sensitive dependence: change one tiny probability by 1%, and
watch the entire outcome distribution rearrange itself.

## How it works

Simulus is not a chatbot and it is not a wrapper around an LLM. It is a custom-built
probabilistic reasoning engine with four distinct stages:

### 1. Situation Parser

Natural language processing extracts the key elements from your input: who is involved,
what is at stake, the environment, and the emotional context. This uses spaCy for
dependency parsing and entity extraction.

### 2. Causal Graph Builder

A directed acyclic graph is constructed where each node is a state of the world and
each edge is a possible action or consequence. The graph branches at decision points
and cascades through 6 levels of cause and effect. Built on networkx.

### 3. Probability Scoring Engine

Each branch in the graph is assigned a conditional probability using a parameterized
scoring model with domain-specific priors, first-order Markov transitions, and
modifiers for emotion, power dynamics, and stake severity. This is not Bayesian
inference in the textbook sense -- there is no likelihood function or posterior
update -- but a structured, reproducible scoring pipeline.

### 4. Sensitive Dependence Simulator

A Lyapunov-style chaos multiplier is applied at each depth level. Small differences
in initial probabilities are amplified exponentially as they cascade through the tree.
A Monte Carlo simulator runs 10,000 stochastic tree walks to produce statistical
distributions of outcomes.

### ML Domain Classifier (optional)

An optional fine-tuned DistilBERT classifier detects the domain (career, relationship,
health, finance, education, travel) and emotional state of the input. If the ML model
is available and confident, it overrides the keyword-based heuristic. If not, the
system falls back gracefully.

The domain classifier achieves 100% accuracy on the evaluation set. This is expected,
not suspicious: 6 well-separated classes with distinct vocabulary clusters are a
trivially separable problem for a fine-tuned transformer with 66 million parameters.
The harder task is emotion classification (75.5% accuracy on 6 classes), where
linguistic markers overlap heavily across categories. For any input that does not
cleanly map to a specific domain, the system falls back to a "general" domain with
context-derived consequence generation that composes labels from the parsed stakes,
actors, and emotional context.

## Features

**Interactive mode** -- Describe any situation and explore branching futures in a
cinematic terminal interface.

**Replay mode** -- Re-run any simulation with the same seed and get byte-identical
results. Rewind to any decision point and take the other path. See how the
trajectories diverge.

**Butterfly mode** -- Apply a tiny perturbation (0.01 change in one probability) and
compare the two resulting scenario branches side by side. Witness sensitive dependence
in action.

**Seed-reproducible by design** -- Same seed, same input, same scenario tree. Every
time. This is the computational core of the project: given identical initial
conditions and the same random seed, the simulation is fully reproducible. But
the branching output shows multiple possible trajectories, reflecting our epistemic
uncertainty about which path will actually unfold.

## Installation

```
pip install simulus
```

Or from source:

```
git clone https://github.com/qubitalgo/simulus.git
cd simulus
pip install -e .
```

## Usage

### Interactive mode

```
simulus
```

You will be prompted to describe a situation. The engine will generate a decision tree
and animate it in your terminal.

### With a specific scenario

```
simulus --scenario "I am about to quit my job to start a company"
```

### Replay with a seed

```
simulus --scenario "asking for a raise at lunch" --seed 42
```

Run this twice. You will get the exact same scenario tree both times.

### Butterfly mode

```
simulus --scenario "asking for a raise at lunch" --seed 42 --butterfly
```

Watch two nearly identical scenario branches diverge into completely different
outcome distributions.

### Replay alternate track

```
simulus --scenario "asking for a raise at lunch" --seed 42 --rewind 2
```

Rewind to decision point 2 and take the path not taken.

## Dependencies

- Python 3.10+
- spaCy -- natural language parsing
- rich -- terminal rendering
- numpy -- probability computation
- networkx -- causal graph structure
- click -- CLI interface

## The Philosophy

This project is a computational exploration of sensitive dependence on initial
conditions, epistemic uncertainty, and the nature of consequence.

Laplace argued in 1814 that a sufficiently powerful intellect, knowing the position
and momentum of every particle, could predict the entire future. Simulus is a small
step toward that thought experiment -- a system that takes a snapshot of a situation
and extrapolates the consequences across multiple scenario branches.

But it also demonstrates why Laplace's Demon is impossible in practice. Chaos theory
shows that infinitesimally small differences in initial conditions lead to exponentially
divergent outcomes. This sensitive dependence is not a metaphor -- it is a mathematical
property of nonlinear dynamical systems, and Simulus makes it visible.

A note on terminology: the system is *seed-reproducible* (same seed and input always
produce the same output), but it is not "deterministic" in the philosophical sense of
implying a single fixed future. The branching scenario tree represents *epistemic*
uncertainty -- our incomplete knowledge of all the variables -- not multiple ontological
realities. The seed controls the random number generator, making the computation
repeatable, but the output deliberately shows many possible trajectories because no
finite model can collapse the future to a single path.

## Why the Seed?

The system produces a branching tree of scenario outcomes, not a single prediction.
Two questions arise: why does it branch, and why does it need a seed?

### The branching tree models epistemic uncertainty

The tree shows multiple scenario branches not because the system is "unsure" in some
vague sense, but because *you* lack complete information. You do not know every other
person's intentions, every market condition, or your own future psychology. The tree is
a map of the consequence space given what the model can extract from your input. A
single deterministic path would require omniscience.

### The seed pins down the unknowns

The seed controls the random number generator that determines which consequences are
selected, how probabilities are distributed, and what noise is injected at each depth.
Without a seed, every run would produce a different tree -- not because the situation
changed, but because the random sampling changed. That would make the output unreliable
and unanalyzable.

The seed says: *given this exact snapshot of a situation, here is one fully specified
projection of the consequence space.* A different seed produces a different but equally
valid projection -- like two analysts modeling the same scenario with different
assumptions.

### Different seeds are an ensemble

This is analogous to ensemble forecasting in meteorology. A weather model run with
different initial conditions produces a spread of forecasts. No single run is "the"
prediction. The spread tells you how sensitive the outcome is to unknowns. In Simulus,
running the same input with different seeds produces different but equally valid
scenario trees, and the variance between them reflects the model's sensitivity to
micro-level assumptions.

### The seed makes sensitivity analysis rigorous

Butterfly mode applies a tiny perturbation and compares the resulting trajectories. This
only works *because* the seed holds everything else constant. Without it, you could not
distinguish "the outcome changed because I perturbed it" from "the outcome changed
because the random sampling was different." The seed is what makes the sensitive
dependence demonstration isolable and repeatable.

| Concept | Role |
|---------|------|
| Branching tree | Models epistemic uncertainty -- you do not know which path will unfold |
| Seed | Pins down random micro-variables so the projection is reproducible |
| Different seeds | Different equally valid projections of the same scenario |
| Butterfly mode | Only meaningful because the seed holds everything else constant |

## Project Structure

```
simulus/
  core/
    parser.py          -- NLP situation parser
    causal_graph.py    -- Directed acyclic graph of consequences
    bayesian.py        -- Parameterized probability scoring engine
    chaos.py           -- Sensitive dependence / Lyapunov multiplier
    montecarlo.py      -- Monte Carlo stochastic tree walk simulator
  renderer/
    tree.py            -- Animated ASCII tree
    effects.py         -- Visual effects (typewriter, colors, particles)
    rewind.py          -- Split-screen alternate trajectory viewer
  cli.py               -- Main entry point
  seed.py              -- Seed-reproducible random number manager
tests/
```

## License

MIT

## Contributing

Simulus is open source. Contributions are welcome. See the issues tab for things that
need doing.

Keep the code clean, the comments minimal, and the math rigorous.
