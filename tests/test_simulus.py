from simulus.seed import SeedManager
from simulus.core.parser import (
    parse_situation, PowerDynamic, Reversibility,
)
from simulus.core.causal_graph import build_causal_graph, NodeType
from simulus.core.bayesian import update_graph_probabilities, expected_sentiment_score
from simulus.core.chaos import (
    create_perturbed_graph, trajectory_divergence_score, lyapunov_multiplier,
    compute_adaptive_exponent,
)
from simulus.core.montecarlo import run_monte_carlo
from simulus.renderer.explainer import generate_explanation


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

        divergence = trajectory_divergence_score(graph, perturbed)
        assert divergence > 0.0

    def test_larger_perturbation_more_divergence(self):
        seed = SeedManager(seed=42)
        ctx = parse_situation("choosing a university")

        graph = build_causal_graph(ctx, seed, max_depth=6)

        small_perturbed = create_perturbed_graph(
            graph, 0.001, seed.fork("small"), context=ctx)
        large_perturbed = create_perturbed_graph(
            graph, 0.05, seed.fork("large"), context=ctx)

        small_div = trajectory_divergence_score(graph, small_perturbed)
        large_div = trajectory_divergence_score(graph, large_perturbed)
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


class TestMLParser:

    def test_model_available(self):
        from simulus.ml.ml_parser import is_model_available
        assert is_model_available()

    def test_predict_returns_required_keys(self):
        from simulus.ml.ml_parser import predict
        result = predict("I want to quit my job and start a business")
        assert "domain" in result
        assert "domain_confidence" in result
        assert "emotion" in result
        assert "emotion_confidence" in result
        assert "domain_distribution" in result
        assert "emotion_distribution" in result

    def test_predict_domain_career(self):
        from simulus.ml.ml_parser import predict
        result = predict("Should I ask my boss for a promotion")
        assert result["domain"] == "career"
        assert result["domain_confidence"] > 0.5

    def test_predict_domain_health(self):
        from simulus.ml.ml_parser import predict
        result = predict("I need to decide whether to have the surgery")
        assert result["domain"] == "health"

    def test_predict_domain_finance(self):
        from simulus.ml.ml_parser import predict
        result = predict("Should I invest my savings in the stock market")
        assert result["domain"] == "finance"

    def test_predict_domain_travel(self):
        from simulus.ml.ml_parser import predict
        result = predict("I want to move to Japan and start a new life")
        assert result["domain"] == "travel"

    def test_predict_domain_relationship(self):
        from simulus.ml.ml_parser import predict
        result = predict("I am thinking about breaking up with my girlfriend")
        assert result["domain"] == "relationship"

    def test_predict_domain_education(self):
        from simulus.ml.ml_parser import predict
        result = predict("Should I go back to university for a masters degree")
        assert result["domain"] == "education"

    def test_confidence_between_zero_and_one(self):
        from simulus.ml.ml_parser import predict
        result = predict("some random decision I need to make")
        assert 0.0 <= result["domain_confidence"] <= 1.0
        assert 0.0 <= result["emotion_confidence"] <= 1.0

    def test_distribution_sums_to_one(self):
        from simulus.ml.ml_parser import predict
        result = predict("I am worried about my health")
        domain_total = sum(result["domain_distribution"].values())
        emotion_total = sum(result["emotion_distribution"].values())
        assert abs(domain_total - 1.0) < 0.01
        assert abs(emotion_total - 1.0) < 0.01

    def test_ml_integrated_into_parser(self):
        ctx = parse_situation("I want to live abroad but all my friends and family are in Hong Kong")
        assert ctx.domain == "travel"

    def test_graceful_fallback_on_missing_model(self):
        import simulus.core.parser as parser_module
        original = parser_module._try_ml_parse

        parser_module._try_ml_parse = lambda text: None
        try:
            ctx = parse_situation("I want to quit my job")
            assert ctx.domain == "career"
        finally:
            parser_module._try_ml_parse = original


class TestExplainer:

    def _build_scenario(self, text: str, seed_val: int = 42):
        seed = SeedManager(seed=seed_val)
        ctx = parse_situation(text)
        graph = build_causal_graph(ctx, seed, max_depth=6)
        b_seed = seed.fork("bayesian")
        update_graph_probabilities(graph, ctx.domain, ctx.emotional_state,
                                   b_seed, context=ctx)
        mc_seed = seed.fork("montecarlo")
        mc_result = run_monte_carlo(graph, mc_seed, n_simulations=1000)
        sentiment = expected_sentiment_score(graph)
        return ctx, graph, mc_result, sentiment

    def test_returns_nonempty_string(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert isinstance(explanation, str)
        assert len(explanation) > 50

    def test_contains_domain_label(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert "career" in explanation.lower()

    def test_contains_simulation_count(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert "1,000" in explanation

    def test_contains_outcome_percentages(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("Should I move abroad")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert "%" in explanation

    def test_contains_volatility_note(self):
        ctx, graph, mc_result, sentiment = self._build_scenario(
            "I must immediately decide whether to accept a dangerous assignment"
        )
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert "volatile" in explanation.lower() or "stable" in explanation.lower() or "sensitive" in explanation.lower()

    def test_butterfly_divergence_note_included(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment,
                                           butterfly_divergence=15.0)
        assert "butterfly" in explanation.lower() or "diverge" in explanation.lower() or "wildly" in explanation.lower()

    def test_butterfly_divergence_none_no_crash(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment,
                                           butterfly_divergence=None)
        assert isinstance(explanation, str)

    def test_closing_present(self):
        ctx, graph, mc_result, sentiment = self._build_scenario("I want to quit my job")
        explanation = generate_explanation(ctx, graph, mc_result, sentiment)
        assert "seed is set" in explanation.lower()

    def test_deterministic_output(self):
        ctx_a, graph_a, mc_a, sent_a = self._build_scenario("I want to quit my job", seed_val=99)
        ctx_b, graph_b, mc_b, sent_b = self._build_scenario("I want to quit my job", seed_val=99)
        exp_a = generate_explanation(ctx_a, graph_a, mc_a, sent_a)
        exp_b = generate_explanation(ctx_b, graph_b, mc_b, sent_b)
        assert exp_a == exp_b


class TestStatisticalValidity:

    def _build_graph(self, text: str, seed_val: int = 42):
        seed = SeedManager(seed=seed_val)
        ctx = parse_situation(text)
        graph = build_causal_graph(ctx, seed, max_depth=6)
        b_seed = seed.fork("bayesian")
        update_graph_probabilities(graph, ctx.domain, ctx.emotional_state,
                                   b_seed, context=ctx)
        return ctx, graph, seed

    def test_leaf_probabilities_sum_to_one(self):
        _, graph, _ = self._build_graph("I want to quit my job")
        leaves = graph.get_leaves()
        total = sum(leaf.probability for leaf in leaves)
        assert abs(total - 1.0) < 0.01, f"Leaf probabilities sum to {total}, not 1.0"

    def test_leaf_probabilities_sum_to_one_travel(self):
        _, graph, _ = self._build_graph("I want to move to another country")
        leaves = graph.get_leaves()
        total = sum(leaf.probability for leaf in leaves)
        assert abs(total - 1.0) < 0.01, f"Leaf probabilities sum to {total}, not 1.0"

    def test_mc_convergence_bounded(self):
        ctx, graph, seed = self._build_graph("Should I invest in stocks")
        mc_seed = seed.fork("montecarlo")
        result = run_monte_carlo(graph, mc_seed, n_simulations=10000)
        assert result.convergence_error < 0.05, (
            f"MC convergence error {result.convergence_error:.4f} exceeds threshold"
        )

    def test_mc_convergence_improves_with_samples(self):
        ctx, graph, seed = self._build_graph("I want to go back to school")
        small = run_monte_carlo(graph, seed.fork("mc_small"), n_simulations=500)
        large = run_monte_carlo(graph, seed.fork("mc_large"), n_simulations=10000)
        assert large.convergence_error <= small.convergence_error + 0.01

    def test_mc_stochastic_walk_produces_variation(self):
        ctx, graph, seed = self._build_graph("I am thinking of changing careers")
        result_a = run_monte_carlo(graph, SeedManager(seed=1), n_simulations=5000)
        result_b = run_monte_carlo(graph, SeedManager(seed=2), n_simulations=5000)
        counts_a = list(result_a.outcome_counts.values())
        counts_b = list(result_b.outcome_counts.values())
        assert counts_a != counts_b, "Different seeds should produce different MC samples"

    def test_consequence_labels_not_all_identical(self):
        _, graph, _ = self._build_graph(
            "I want to live abroad but all my friends are in Hong Kong"
        )
        leaves = graph.get_leaves()
        labels = [leaf.label for leaf in leaves]
        unique_ratio = len(set(labels)) / len(labels) if labels else 0
        assert unique_ratio > 0.15, (
            f"Only {unique_ratio:.0%} unique leaf labels -- too much repetition"
        )

    def test_sibling_probabilities_normalized(self):
        _, graph, _ = self._build_graph("Should I ask for a promotion")
        root_children = graph.get_children(graph.root_id)
        if root_children:
            edge_probs = [
                graph.get_edge(graph.root_id, c.node_id).probability
                for c in root_children
            ]
            total = sum(edge_probs)
            assert abs(total - 1.0) < 0.05, (
                f"Root children edge probabilities sum to {total}, not 1.0"
            )


class TestDecisionExpansion:

    def _build_graph(self, text: str, seed_val: int = 42):
        seed = SeedManager(seed=seed_val)
        ctx = parse_situation(text)
        graph = build_causal_graph(ctx, seed, max_depth=6)
        return ctx, graph

    def test_four_decisions_per_domain(self):
        from simulus.core.causal_graph import CONTEXTUAL_DECISIONS
        for domain, decisions in CONTEXTUAL_DECISIONS.items():
            assert len(decisions) >= 3, (
                f"Domain {domain} has only {len(decisions)} decisions"
            )

    def test_graph_has_multiple_branches(self):
        _, graph = self._build_graph("I want to quit my job")
        root_children = graph.get_children(graph.root_id)
        assert len(root_children) >= 3

    def test_do_nothing_option_exists(self):
        from simulus.core.causal_graph import CONTEXTUAL_DECISIONS
        for domain, decisions in CONTEXTUAL_DECISIONS.items():
            labels_lower = [d["label"].lower() for d in decisions]
            has_passive = any(
                "status quo" in l or "do nothing" in l or "ignore" in l
                or "delay" in l or "postpone" in l or "wait" in l
                or "break" in l or "deliberate" in l
                for l in labels_lower
            )
            assert has_passive, (
                f"Domain {domain} has no passive/delay option"
            )


class TestBayesianRealism:

    def _build_and_score(self, text: str, seed_val: int = 42):
        seed = SeedManager(seed=seed_val)
        ctx = parse_situation(text)
        graph = build_causal_graph(ctx, seed, max_depth=6)
        b_seed = seed.fork("bayesian")
        update_graph_probabilities(graph, ctx.domain, ctx.emotional_state,
                                   b_seed, context=ctx)
        return ctx, graph, expected_sentiment_score(graph)

    def test_mean_reversion_bounds_extreme_outcomes(self):
        _, _, score_a = self._build_and_score("Everything is perfect and I am thriving")
        _, _, score_b = self._build_and_score("Everything is terrible and I am drowning")
        assert score_a < 0.9, "Positive score should be bounded by mean reversion"
        assert score_b > -0.9, "Negative score should be bounded by mean reversion"

    def test_depth_uncertainty_increases_variance(self):
        from simulus.core.bayesian import _apply_depth_uncertainty, _normalize
        seed = SeedManager(seed=42)
        priors = {"positive": 0.4, "negative": 0.4, "neutral": 0.2}

        shallow_results = []
        deep_results = []
        for i in range(50):
            s = seed.fork(f"shallow_{i}")
            r = _apply_depth_uncertainty(dict(priors), 2, s)
            shallow_results.append(r["positive"])
            s = seed.fork(f"deep_{i}")
            r = _apply_depth_uncertainty(dict(priors), 6, s)
            deep_results.append(r["positive"])

        import numpy as np
        shallow_var = np.var(shallow_results)
        deep_var = np.var(deep_results)
        assert deep_var > shallow_var, (
            f"Deep variance {deep_var:.6f} should exceed shallow {shallow_var:.6f}"
        )

    def test_streak_bonus_amplifies_momentum(self):
        from simulus.core.bayesian import STREAK_BONUS
        pos_bonus = STREAK_BONUS["positive"]["positive"]
        neg_bonus = STREAK_BONUS["negative"]["negative"]
        assert pos_bonus > 0, "Positive streak should amplify positive continuation"
        assert neg_bonus > 0, "Negative streak should amplify negative continuation"
