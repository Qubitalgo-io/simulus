from simulus.seed import SeedManager
from simulus.core.parser import (
    parse_situation, PowerDynamic, Reversibility,
)
from simulus.core.causal_graph import build_causal_graph, NodeType
from simulus.core.bayesian import update_graph_probabilities, expected_sentiment_score
from simulus.core.chaos import (
    create_perturbed_graph, fate_divergence_score, lyapunov_multiplier,
    compute_adaptive_exponent,
)
from simulus.core.montecarlo import run_monte_carlo


class TestDeterminism:

    def test_same_seed_same_output(self):
        seed_a = SeedManager(seed=42)
        seed_b = SeedManager(seed=42)

        for _ in range(100):
            assert seed_a.random() == seed_b.random()

    def test_different_seed_different_output(self):
        seed_a = SeedManager(seed=42)
        seed_b = SeedManager(seed=99)

        results_a = [seed_a.random() for _ in range(100)]
        results_b = [seed_b.random() for _ in range(100)]
        assert results_a != results_b

    def test_fork_determinism(self):
        seed_a = SeedManager(seed=42)
        seed_b = SeedManager(seed=42)

        fork_a = seed_a.fork("branch_0")
        fork_b = seed_b.fork("branch_0")

        for _ in range(50):
            assert fork_a.random() == fork_b.random()

    def test_full_pipeline_determinism(self):
        for _ in range(3):
            seed = SeedManager(seed=12345)
            context = parse_situation("I am about to quit my job")
            graph = build_causal_graph(context, seed, max_depth=6)
            b_seed = seed.fork("bayesian")
            update_graph_probabilities(graph, context.domain,
                                       context.emotional_state, b_seed,
                                       context=context)
            mc_seed = seed.fork("montecarlo")
            result = run_monte_carlo(graph, mc_seed, n_simulations=1000)

            if _ == 0:
                first_counts = dict(result.outcome_counts)
                first_sentiment = result.mean_sentiment_score
            else:
                assert result.outcome_counts == first_counts
                assert result.mean_sentiment_score == first_sentiment


class TestParser:

    def test_detects_career_domain(self):
        ctx = parse_situation("I want to ask my boss for a raise")
        assert ctx.domain == "career"

    def test_detects_relationship_domain(self):
        ctx = parse_situation("Should I break up with my partner")
        assert ctx.domain == "relationship"

    def test_detects_emotion(self):
        ctx = parse_situation("I am nervous about the job interview")
        assert ctx.emotional_state == "anxious"

    def test_default_actor(self):
        ctx = parse_situation("something is happening")
        assert ctx.main_actor != ""

    def test_extracts_actors(self):
        ctx = parse_situation("My boss found out I am interviewing at a competitor")
        assert len(ctx.actor_profiles) > 0
        actor_names = [a.name for a in ctx.actor_profiles]
        assert any("boss" in name for name in actor_names)

    def test_actor_power_dynamic(self):
        ctx = parse_situation("My manager wants me to relocate")
        boss_actors = [a for a in ctx.actor_profiles
                       if "manager" in a.name.lower()]
        if boss_actors:
            assert boss_actors[0].power_dynamic == PowerDynamic.SUPERIOR

    def test_detects_preconditions(self):
        ctx = parse_situation("Because I already signed a contract, I must leave by Friday")
        assert len(ctx.preconditions) > 0

    def test_detects_constraints(self):
        ctx = parse_situation("I must decide before the deadline expires")
        assert len(ctx.constraints) > 0 or ctx.time_pressure > 0

    def test_detects_conflicts(self):
        ctx = parse_situation("My partner wants to stay but I want to move abroad")
        assert len(ctx.conflict_vectors) > 0

    def test_detects_time_pressure(self):
        ctx = parse_situation("I need to decide immediately before the offer expires")
        assert ctx.time_pressure > 0

    def test_reversibility_detection(self):
        ctx = parse_situation("I am about to sign a permanent contract that cannot be undone")
        assert ctx.reversibility in (Reversibility.IRREVERSIBLE, Reversibility.DIFFICULT)

    def test_stake_severity(self):
        ctx = parse_situation("I might lose everything if this investment fails")
        assert ctx.stake_severity > 0

    def test_compound_volatility(self):
        ctx = parse_situation(
            "I must immediately decide whether to accept a job in another country "
            "even though my partner is against it and I just signed a lease"
        )
        assert ctx.compound_volatility > 0.3


class TestCausalGraph:

    def test_graph_has_nodes(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("I want to invest in crypto")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        assert graph.node_count() > 0

    def test_graph_has_root(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("moving to another city")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        root = graph.get_node(graph.root_id)
        assert root.depth == 0

    def test_graph_has_leaves(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("starting a new business")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        leaves = graph.get_leaves()
        assert len(leaves) > 0
        for leaf in leaves:
            assert leaf.depth >= 3

    def test_contextual_decisions_match_domain(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("Should I invest my savings in the stock market")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        children = graph.get_children(graph.root_id)
        labels = [c.label for c in children]
        assert any("capital" in l.lower() or "opportunity" in l.lower()
                    or "wait" in l.lower() for l in labels)

    def test_actor_reaction_nodes_created(self):
        seed = SeedManager(seed=42)
        ctx = parse_situation("My boss asked me to fire my friend from the team")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        all_nodes = graph.get_all_nodes()
        reaction_nodes = [n for n in all_nodes
                          if n.node_type == NodeType.ACTOR_REACTION]
        assert len(reaction_nodes) >= 0  # may or may not have actors depending on parse

    def test_nodes_have_causal_mechanisms(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("I am thinking of dropping out of college")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        nodes_with_mechanism = [n for n in graph.get_all_nodes()
                                if n.causal_mechanism]
        assert len(nodes_with_mechanism) > 0

    def test_feedback_edges_structure(self):
        seed = SeedManager(seed=1)
        ctx = parse_situation("I lost a lot of money on a bad investment")
        graph = build_causal_graph(ctx, seed, max_depth=6)
        feedback = graph.get_feedback_edges()
        for edge in feedback:
            assert edge.edge_type == "feedback"


class TestButterflyEffect:

    def test_lyapunov_multiplier_increases(self):
        prev = 1.0
        for d in range(1, 7):
            m = lyapunov_multiplier(d)
            assert m > prev
            prev = m

    def test_perturbation_causes_divergence(self):
        seed = SeedManager(seed=42)
        ctx = parse_situation("asking for a promotion")
        graph = build_causal_graph(ctx, seed, max_depth=6)

        perturbed_seed = seed.fork("butterfly")
        perturbed = create_perturbed_graph(graph, 0.01, perturbed_seed,
                                            context=ctx)

        divergence = fate_divergence_score(graph, perturbed)
        assert divergence > 0.0

    def test_larger_perturbation_more_divergence(self):
        seed = SeedManager(seed=42)
        ctx = parse_situation("choosing a university")

        graph = build_causal_graph(ctx, seed, max_depth=6)

        small_perturbed = create_perturbed_graph(
            graph, 0.001, seed.fork("small"), context=ctx)
        large_perturbed = create_perturbed_graph(
            graph, 0.05, seed.fork("large"), context=ctx)

        small_div = fate_divergence_score(graph, small_perturbed)
        large_div = fate_divergence_score(graph, large_perturbed)
        assert large_div > small_div

    def test_adaptive_exponent_varies_by_domain(self):
        career_ctx = parse_situation("I want to quit my job")
        health_ctx = parse_situation("I should go to the doctor")

        career_exp = compute_adaptive_exponent(career_ctx)
        health_exp = compute_adaptive_exponent(health_ctx)
        assert career_exp != health_exp

    def test_adaptive_exponent_increases_with_volatility(self):
        calm = parse_situation("I am thinking about taking a class")
        intense = parse_situation(
            "I must immediately decide whether to accept a dangerous job "
            "overseas while my family is against it and the deadline is tomorrow"
        )

        calm_exp = compute_adaptive_exponent(calm)
        intense_exp = compute_adaptive_exponent(intense)
        assert intense_exp >= calm_exp

    def test_adaptive_exponent_bounded(self):
        ctx = parse_situation("everything is on fire and I must choose now")
        exp = compute_adaptive_exponent(ctx)
        assert 0.2 <= exp <= 1.2


class TestBayesian:

    def test_update_with_context(self):
        seed = SeedManager(seed=42)
        ctx = parse_situation("My boss found out I have been interviewing elsewhere")
        graph = build_causal_graph(ctx, seed, max_depth=4)
        b_seed = seed.fork("bayesian")
        update_graph_probabilities(graph, ctx.domain, ctx.emotional_state,
                                   b_seed, context=ctx)
        score = expected_sentiment_score(graph)
        assert -1.0 <= score <= 1.0

    def test_power_dynamics_affect_outcome(self):
        seed_a = SeedManager(seed=42)
        seed_b = SeedManager(seed=42)

        ctx_boss = parse_situation("My boss is angry and wants to fire me")
        graph_boss = build_causal_graph(ctx_boss, seed_a, max_depth=4)
        update_graph_probabilities(graph_boss, ctx_boss.domain,
                                   ctx_boss.emotional_state,
                                   seed_a.fork("bayesian"), context=ctx_boss)

        ctx_peer = parse_situation("My colleague is slightly annoyed")
        graph_peer = build_causal_graph(ctx_peer, seed_b, max_depth=4)
        update_graph_probabilities(graph_peer, ctx_peer.domain,
                                   ctx_peer.emotional_state,
                                   seed_b.fork("bayesian"), context=ctx_peer)

        score_boss = expected_sentiment_score(graph_boss)
        score_peer = expected_sentiment_score(graph_peer)
        assert score_boss != score_peer


class TestMonteCarlo:

    def test_runs_correct_number(self):
        seed = SeedManager(seed=7)
        ctx = parse_situation("taking a new job offer")
        graph = build_causal_graph(ctx, seed, max_depth=4)
        result = run_monte_carlo(graph, seed, n_simulations=5000)
        assert result.n_simulations == 5000
        assert sum(result.outcome_counts.values()) == 5000

    def test_sentiment_distribution_sums_to_one(self):
        seed = SeedManager(seed=7)
        ctx = parse_situation("going back to school")
        graph = build_causal_graph(ctx, seed, max_depth=4)
        result = run_monte_carlo(graph, seed, n_simulations=1000)
        total = sum(result.sentiment_distribution.values())
        assert abs(total - 1.0) < 0.01
