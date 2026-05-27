import pytest
from app.services.report_agent import (
    SourceCredibilityScorer,
    ContradictionDetector,
    NumericalSanityChecker,
    SpecificityDetector,
    ConfidenceCoverageChecker,
    AnalyticalReport,
    GroundingReport,
    ReportAgent,
    EvidenceAlignmentChecker,
    AnchoringDistributionChecker,
    CausalCompletenessChecker,
    NumericalGroundingCoverageChecker
)
from app.services.analytical_modules.markets import MarketsAnalyticalModule
from app.services.analytical_modules.policy import PolicyAnalyticalModule
from app.services.analytical_modules.organizational_risk import OrganizationalRiskAnalyticalModule
from app.services.analytical_modules import BaseAnalyticalModule

def test_source_credibility_scorer():
    scorer = SourceCredibilityScorer()
    draft = "Stock price rose rapidly [S1]. The market jumped [S2]."
    sources_metadata = [
        (1, "interview_agents", "Agents think stocks are up"),
        (2, "web_search", "Stocks hit high")
    ]
    report = scorer.check(draft, sources_metadata)
    assert report.source_weights[1] == "HIGH"
    assert report.source_weights[2] == "LOW"
    assert "S1:HIGH" in report.summary
    assert "S2:LOW" in report.summary
    # Since only single citations are in each sentence, cross_referenced_claims should be empty
    assert len(report.cross_referenced_claims) == 0

    # Test sentence with multiple citations
    draft_multi = "Sentiment has surged significantly [S1] [S2]."
    report_multi = scorer.check(draft_multi, sources_metadata)
    assert len(report_multi.cross_referenced_claims) == 1
    assert "Sentiment has surged" in report_multi.cross_referenced_claims[0]


def test_contradiction_detector_finds_contradictions():
    detector = ContradictionDetector()
    draft = "CompanyX stock price rose quickly [S1]. However, CompanyX declined sharply afterward [S2]."
    report = detector.check(draft)
    assert len(report.contradictions) >= 1
    assert report.contradictions[0][0] == "CompanyX"
    assert report.contradictions[0][1] == 1
    assert report.contradictions[0][2] == 2
    assert "Sentence A" in report.contradictions[0][3]
    assert report.warning is not None


def test_contradiction_detector_ignores_consistent_sentences():
    detector = ContradictionDetector()
    draft = "CompanyX stock price rose quickly [S1]. CompanyX support is growing [S2]."
    report = detector.check(draft)
    assert len(report.contradictions) == 0
    assert report.warning is None


def test_numerical_sanity_checker_detects_anomalies():
    checker = NumericalSanityChecker()
    # 1. Proportional percentage > 100%
    draft_pct = "The public support reached 150% in the latest surveys [S1]."
    report_pct = checker.check(draft_pct, [])
    assert len(report_pct.anomalies) >= 1
    assert "exceeds 100%" in report_pct.anomalies[0]

    # 2. Percentage breakdown sum anomaly
    draft_sum = "The division splits as 60% supportive, 30% opposing, and 40% neutral [S1]."
    report_sum = checker.check(draft_sum, [])
    assert len(report_sum.anomalies) >= 1
    assert "sum to" in report_sum.anomalies[0]


def test_numerical_sanity_checker_detects_order_of_magnitude_mismatch():
    checker = NumericalSanityChecker()
    draft = "The market capitalization reached 12 billion [S1]."
    sources_metadata = [
        (1, "quick_search", "The valuation was estimated at 12 million dollars.")
    ]
    report = checker.check(draft, sources_metadata)
    assert len(report.anomalies) >= 1
    assert "order of magnitude mismatch" in report.anomalies[0]


def test_specificity_detector():
    detector = SpecificityDetector()
    
    # Sentence with vague phrase and no source
    draft_vague = "We must monitor closely going forward."
    report_vague = detector.check(draft_vague)
    assert len(report_vague.vague_sentences) >= 1
    assert report_vague.warning is not None

    # Sentence with vague phrase but has source
    draft_backed = "We must monitor closely going forward [S1]."
    report_backed = detector.check(draft_backed)
    assert len(report_backed.vague_sentences) == 0


def test_confidence_coverage_checker():
    checker = ConfidenceCoverageChecker()

    # Speculative sentence without confidence label
    draft_spec = "This suggests that stock prices could fluctuate tomorrow."
    report_spec = checker.check(draft_spec)
    assert len(report_spec.uncalibrated_speculations) >= 1
    assert "speculative claims lack confidence labels" in report_spec.warning

    # Speculative sentence with confidence label
    draft_calibrated = "This suggests with low confidence that stock prices could fluctuate."
    report_calibrated = checker.check(draft_calibrated)
    assert len(report_calibrated.uncalibrated_speculations) == 0

    # Confident claim without source
    draft_overconfident = "This will definitely happen."
    report_overconfident = checker.check(draft_overconfident)
    assert len(report_overconfident.overconfident_unbacked_claims) >= 1
    assert "confident claims lack source citations" in report_overconfident.warning


def test_domain_modules_integration():
    markets_module = MarketsAnalyticalModule()
    policy_module = PolicyAnalyticalModule()
    risk_module = OrganizationalRiskAnalyticalModule()

    # Test vague phrases
    assert "market will decide" in markets_module.get_vague_phrases()
    assert "policy changes are likely" in policy_module.get_vague_phrases()
    assert "organizational challenges" in risk_module.get_vague_phrases()

    # Test markets numerical sanity
    anom_markets = markets_module.check_numerical_sanity("Company has P/E ratio of 1500", [])
    assert len(anom_markets) >= 1
    assert "P/E" in anom_markets[0]

    # Test policy numerical sanity (ancient act)
    anom_policy = policy_module.check_numerical_sanity("According to the Act of 1840, regulations apply", [])
    assert len(anom_policy) >= 1
    assert "1840" in anom_policy[0]

    # Test risk severity scale out of bounds
    anom_risk = risk_module.check_numerical_sanity("Operational severity score of 12/10", [])
    assert len(anom_risk) >= 1
    assert "severity" in anom_risk[0]


def test_report_agent_module_selection():
    agent = ReportAgent("dummy_graph", "dummy_sim", "Dormitory formaldehyde policy regulation and risk")
    module = agent._select_analytical_module("Market reactions to policy")
    # Both "policy" and "market" exist. "market" comes first in our cascade, so it selects Markets
    assert isinstance(module, MarketsAnalyticalModule)

    # Test policy only
    agent_p = ReportAgent("dummy_graph", "dummy_sim", "New compliance and enforcement act")
    module_p = agent_p._select_analytical_module("Regulatory review section")
    assert isinstance(module_p, PolicyAnalyticalModule)

    # Test operational risk only
    agent_r = ReportAgent("dummy_graph", "dummy_sim", "Operational continuity under crisis")
    module_r = agent_r._select_analytical_module("Vulnerability assessment")
    assert isinstance(module_r, OrganizationalRiskAnalyticalModule)

    # Test generic fallback
    agent_f = ReportAgent("dummy_graph", "dummy_sim", "Some general topic")
    module_f = agent_f._select_analytical_module("General discussion")
    assert isinstance(module_f, BaseAnalyticalModule)


def test_evidence_evaluator_and_digest():
    from app.services.evidence_evaluator import EvidenceEvaluator, EvidenceScore
    evaluator = EvidenceEvaluator()

    # 1. Score evidence
    score = evaluator.score_evidence(
        source_num=1,
        raw_result="The quarterly revenue plunged 15% due to regulatory compliance cost of $5 million USD.",
        tool_name="insight_forge",
        section_title="Financial and market impact",
        simulation_requirement="Simulate formaldehyde compliance cost impact on market valuation"
    )
    assert score.source_num == 1
    assert score.tool_name == "insight_forge"
    assert score.credibility >= 0.9
    assert score.epistemic_tier == "SIMULATION_PRIMARY"
    assert score.strategic_materiality > 0.5  # contains revenue, cost, compliance cost, $5 million USD
    assert score.quantified_claims_count >= 1

    # 2. Build evidence card
    card = evaluator.build_evidence_card(score, "Raw Result Content")
    assert "[Evidence Card — insight_forge — S1]" in card
    assert "Weight:" in card
    assert "Raw Result Content" in card

    # 3. Detect concentration (needs >=3 sources)
    score1 = score
    score2 = evaluator.score_evidence(
        source_num=2,
        raw_result="A brief statement with low specificity and no metrics.",
        tool_name="web_search",
        section_title="Financial and market impact",
        simulation_requirement="Simulate formaldehyde compliance cost impact on market valuation",
        freshness_score=0.3
    )
    score3 = evaluator.score_evidence(
        source_num=3,
        raw_result="Another short phrase.",
        tool_name="quick_search",
        section_title="Financial and market impact",
        simulation_requirement="Simulate formaldehyde compliance cost impact on market valuation",
        freshness_score=0.4
    )
    # Since score1 is insight_forge and highly strategic/material, its weight will be much higher than score2 and score3 (both low/medium)
    all_scores = [score1, score2, score3]
    warning = evaluator.detect_concentration(all_scores)
    assert warning is not None
    assert "[Evidence Concentration Warning]" in warning

    # 4. Build digest
    digest = evaluator.build_digest(all_scores)
    assert "EPISTEMIC HIERARCHY" in digest
    assert "[S1]" in digest and "insight_forge" in digest


def test_skepticism_checker():
    from app.services.report_agent import SkepticismChecker
    from app.services.evidence_evaluator import EvidenceScore
    checker = SkepticismChecker()

    # Fake scored sources map
    evidence_scores = {
        1: EvidenceScore(
            source_num=1, tool_name="web_search", credibility=0.4, relevance=0.8, recency=0.9, 
            specificity=0.3, strategic_materiality=0.2, synthesis_weight=0.35,
            key_claims_count=0, quantified_claims_count=0, speculative_claims_count=0
        )
    }

    # Draft with overstated certainty, evidence mismatch (citing low-credibility S1 as definitive fact), and missing alternatives
    draft = "We will definitely dominate the market [S1]. The pricing is always perfect and will surely rise [S1]."
    report = checker.check(draft, evidence_scores)
    
    assert report.overstated_certainty is True
    assert len(report.mismatches) >= 1
    assert "uses definitive language" in report.mismatches[0]
    assert report.missing_alternatives is True
    assert report.warning is not None
    assert "SKEPTICISM WARNING" in report.warning


def test_markets_quantitative_grounding():
    module = MarketsAnalyticalModule()
    
    # Financial terms mentioned but absolutely no quantitative figures
    anoms = module.check_quantitative_grounding("Our revenue, valuation, and profits have changed.", [])
    assert len(anoms) >= 1
    assert "revenue, valuation, or earnings" in anoms[0]

    # Grounding is present, should pass
    anoms_ok = module.check_quantitative_grounding("Our revenue increased to $15 million USD, representing a 20% margin.", [])
    assert len(anoms_ok) == 0


def test_evidence_alignment_checker_numeric_mismatch():
    from app.services.evidence_evaluator import EvidenceScore

    checker = EvidenceAlignmentChecker()

    sources_metadata = [
        (1, "web_search", "The quarterly revenue rose 5% to $10 million in 2024."),
    ]
    evidence_scores = {
        1: EvidenceScore(
            source_num=1,
            tool_name="web_search",
            credibility=0.4,
            relevance=0.8,
            recency=0.9,
            specificity=0.3,
            strategic_materiality=0.2,
            synthesis_weight=0.35,
            key_claims_count=1,
            quantified_claims_count=1,
            speculative_claims_count=0,
        )
    }

    # Numeric claim intentionally mismatches both % and $ amount.
    draft = "The quarterly revenue rose 15% to $12 million in 2024 [S1]."
    report = checker.check(draft, sources_metadata, evidence_scores)

    assert report.warning is not None
    assert len(report.unsupported_sentences) >= 1


def test_evidence_alignment_checker_numeric_match():
    from app.services.evidence_evaluator import EvidenceScore

    checker = EvidenceAlignmentChecker()

    sources_metadata = [
        (1, "web_search", "The quarterly revenue rose 5% to $10 million in 2024."),
    ]
    evidence_scores = {
        1: EvidenceScore(
            source_num=1,
            tool_name="web_search",
            credibility=0.9,
            relevance=0.8,
            recency=0.9,
            specificity=0.8,
            strategic_materiality=0.6,
            synthesis_weight=0.8,
            key_claims_count=1,
            quantified_claims_count=1,
            speculative_claims_count=0,
        )
    }

    draft = "The quarterly revenue rose 5% to $10 million in 2024 [S1]."
    report = checker.check(draft, sources_metadata, evidence_scores)

    assert report.warning is None
    assert len(report.unsupported_sentences) == 0


def test_anchoring_distribution_checker_flags_single_source_dominance():
    from app.services.evidence_evaluator import EvidenceScore

    checker = AnchoringDistributionChecker()

    evidence_scores = {
        1: EvidenceScore(
            source_num=1,
            tool_name="web_search",
            credibility=0.4,
            relevance=0.8,
            recency=0.9,
            specificity=0.3,
            strategic_materiality=0.2,
            synthesis_weight=0.35,
            key_claims_count=1,
            quantified_claims_count=0,
            speculative_claims_count=0,
        ),
        2: EvidenceScore(
            source_num=2,
            tool_name="quick_search",
            credibility=0.7,
            relevance=0.8,
            recency=0.9,
            specificity=0.6,
            strategic_materiality=0.3,
            synthesis_weight=0.5,
            key_claims_count=1,
            quantified_claims_count=0,
            speculative_claims_count=0,
        ),
        3: EvidenceScore(
            source_num=3,
            tool_name="panorama_search",
            credibility=0.6,
            relevance=0.8,
            recency=0.9,
            specificity=0.6,
            strategic_materiality=0.3,
            synthesis_weight=0.55,
            key_claims_count=1,
            quantified_claims_count=0,
            speculative_claims_count=0,
        ),
    }

    draft = (
        "Event A will drive outcomes [S1]. "
        "Event A changes costs [S1]. "
        "Event A affects pricing [S1]. "
        "Event A impacts demand [S1]. "
        "Event A shifts valuation [S1]. "
        "Supporting detail from S2 [S2]. "
        "Supporting detail from S3 [S3]."
    )

    report = checker.check(draft, evidence_scores)
    assert report.warning is not None
    assert "NARRATIVE ANCHORING RISK" in report.warning
    assert "[S1]" in report.warning


def test_causal_completeness_checker_flags_missing_mechanism():
    checker = CausalCompletenessChecker()

    draft = "Revenue will increase and valuation will rise [S1]."
    report = checker.check(draft, "Market valuation impact", "Market reactions to policy")

    assert report.warning is not None
    assert "mechanism" in report.warning.lower()


def test_numerical_grounding_coverage_checker_flags_missing_probability_weighting():
    checker = NumericalGroundingCoverageChecker()

    draft = "Valuation increases from 10x to 12x based on market signals [S1]."
    report = checker.check(draft, "Market valuation impact", "Market reactions to policy")

    assert report.warning is not None
    assert "probability-weighted" in report.warning.lower()


def test_epistemic_discriminator_classifies_social_and_institutional():
    from app.services.epistemic_discriminator import EpistemicDiscriminator

    disc = EpistemicDiscriminator()

    social = disc.classify(
        "A LinkedIn comment argued the merger will definitely succeed according to users.",
        "web_search",
    )
    assert social.tier in ("SOCIAL_OPINION", "SPECULATIVE", "RETAIL_COMMENTARY")
    assert social.confidence_ceiling == "TENTATIVE_ONLY"

    institutional = disc.classify(
        "SEC filing 10-K shows revenue of $5B https://www.sec.gov/archives/...",
        "web_search",
    )
    assert institutional.tier == "INSTITUTIONAL"
    assert institutional.is_primary_eligible


def test_evidence_evaluator_applies_epistemic_cap():
    from app.services.evidence_evaluator import EvidenceEvaluator

    evaluator = EvidenceEvaluator()
    score = evaluator.score_evidence(
        source_num=1,
        raw_result="Reddit and LinkedIn comments say the stock will moon.",
        tool_name="web_search",
        section_title="Market view",
        simulation_requirement="Valuation impact",
    )
    assert score.epistemic_tier in ("SOCIAL_OPINION", "RETAIL_COMMENTARY", "SPECULATIVE")
    assert score.synthesis_weight <= 0.35
    assert score.is_tentative_only


def test_epistemic_review_checker_flags_collapse():
    from app.services.report_agent import EpistemicReviewChecker
    from app.services.evidence_evaluator import EvidenceScore

    checker = EpistemicReviewChecker()
    evidence_scores = {
        1: EvidenceScore(
            source_num=1, tool_name="web_search", credibility=0.2, relevance=0.9,
            recency=0.9, specificity=0.3, strategic_materiality=0.1, synthesis_weight=0.2,
            key_claims_count=1, quantified_claims_count=0, speculative_claims_count=1,
            epistemic_tier="SOCIAL_OPINION", confidence_ceiling="TENTATIVE_ONLY",
            epistemic_label="Social / comment opinion",
        ),
        2: EvidenceScore(
            source_num=2, tool_name="web_search", credibility=0.25, relevance=0.8,
            recency=0.9, specificity=0.3, strategic_materiality=0.1, synthesis_weight=0.22,
            key_claims_count=1, quantified_claims_count=0, speculative_claims_count=0,
            epistemic_tier="RETAIL_COMMENTARY", confidence_ceiling="TENTATIVE_ONLY",
            epistemic_label="Retail financial commentary",
        ),
    }
    draft = (
        "The market will definitely rally [S1]. Analysts upgraded the name [S1]. "
        "LinkedIn sentiment proves bullishness [S2]. Retail blogs confirm upside [S2]."
    )
    report = checker.check(draft, evidence_scores)
    assert report.warning is not None
    assert "EPISTEMIC" in report.warning


def test_section_confidence_profile_downgrades_tentative_heavy_evidence():
    from app.services.evidence_evaluator import EvidenceEvaluator, EvidenceScore

    evaluator = EvidenceEvaluator()
    scores = [
        EvidenceScore(
            source_num=1, tool_name="web_search", credibility=0.2, relevance=0.9,
            recency=0.9, specificity=0.3, strategic_materiality=0.1, synthesis_weight=0.25,
            key_claims_count=1, quantified_claims_count=0, speculative_claims_count=1,
            epistemic_tier="SOCIAL_OPINION", confidence_ceiling="TENTATIVE_ONLY",
            epistemic_label="Social",
        ),
        EvidenceScore(
            source_num=2, tool_name="web_search", credibility=0.25, relevance=0.8,
            recency=0.9, specificity=0.3, strategic_materiality=0.1, synthesis_weight=0.22,
            key_claims_count=1, quantified_claims_count=0, speculative_claims_count=0,
            epistemic_tier="RETAIL_COMMENTARY", confidence_ceiling="TENTATIVE_ONLY",
            epistemic_label="Retail",
        ),
    ]
    profile = evaluator.compute_section_profile(scores)
    assert profile.section_ceiling == "TENTATIVE_ONLY"
    assert profile.assertion_strength == "exploratory"
    block = evaluator.build_confidence_governed_block(profile)
    assert "Banned phrasing" in block
    allocation = evaluator.build_narrative_weight_allocation(scores)
    assert "context-only" in allocation


def test_analytical_report_requires_confidence_rewrite_on_severity():
    from app.services.report_agent import (
        AnalyticalReport,
        GroundingReport,
        CredibilityReport,
        ContradictionReport,
        NumericalSanityReport,
        SpecificityReport,
        ConfidenceReport,
        EvidenceAlignmentReport,
        AnchoringDistributionReport,
        CausalCompletenessReport,
        NumericalGroundingCoverageReport,
        SkepticismReport,
        EpistemicReviewReport,
    )

    def _empty_report(**kwargs):
        defaults = dict(
            grounding=GroundingReport(
                total_sources=1,
                cited_sources=[1],
                uncited_sources=[],
                grounding_score=1.0,
                high_risk_sentences=[],
                warning=None,
                summary="ok",
            ),
            credibility=CredibilityReport(
                source_weights={1: "HIGH"},
                cross_referenced_claims=[],
                summary="ok",
            ),
            contradictions=ContradictionReport(
                contradictions=[], warning=None, summary="ok"
            ),
            numerical_sanity=NumericalSanityReport(
                anomalies=[], warning=None, summary="ok"
            ),
            specificity=SpecificityReport(
                vague_sentences=[], warning=None, summary="ok"
            ),
            confidence=ConfidenceReport(
                uncalibrated_speculations=[],
                overconfident_unbacked_claims=[],
                warning=None,
                summary="ok",
            ),
            evidence_alignment=EvidenceAlignmentReport(
                unsupported_sentences=[],
                alignment_score=1.0,
                warning=None,
                summary="ok",
            ),
            anchoring=AnchoringDistributionReport(warning=None, summary="ok"),
            causal_completeness=CausalCompletenessReport(
                missing_components=[], warning=None, summary="ok"
            ),
            numerical_grounding_coverage=NumericalGroundingCoverageReport(
                missing_components=[], warning=None, summary="ok"
            ),
        )
        defaults.update(kwargs)
        return AnalyticalReport(**defaults)

    mild = _empty_report()
    assert not mild.requires_confidence_rewrite()

    severe = _empty_report(
        confidence=ConfidenceReport(
            uncalibrated_speculations=["a", "b"],
            overconfident_unbacked_claims=["c"],
            summary="bad",
            warning="⚠️ confidence",
        ),
        skepticism=SkepticismReport(
            overstated_certainty=True,
            mismatches=["m1"],
            missing_alternatives=True,
            summary="bad",
            warning="⚠️ skepticism",
        ),
        epistemic_review=EpistemicReviewReport(
            evidence_quality_collapse=True,
            overconfident_low_tier_sentences=["x"],
            missing_primary_anchor=True,
            low_tier_citation_share=0.6,
            critique_questions=[],
            summary="bad",
            warning="⚠️ EPISTEMIC",
        ),
    )
    assert severe.requires_confidence_rewrite()
    assert severe.severity_score() >= 3


def test_evidence_evaluator_durability_and_synthesis_mandate():
    from app.services.evidence_evaluator import EvidenceEvaluator

    evaluator = EvidenceEvaluator()

    durable = evaluator.score_evidence(
        source_num=1,
        raw_result="Structural competitive moat and long-term unit economics drive revenue and margin.",
        tool_name="insight_forge",
        section_title="Market structure",
        simulation_requirement="Market valuation under regulatory change",
    )
    ephemeral = evaluator.score_evidence(
        source_num=2,
        raw_result="Breaking: merger talks reportedly spike today according to sources.",
        tool_name="web_search",
        section_title="Market structure",
        simulation_requirement="Market valuation under regulatory change",
        freshness_score=0.95,
    )

    assert durable.durability > ephemeral.durability
    mandate = evaluator.build_synthesis_mandate([durable, ephemeral], section_title="Test")
    assert "PRE-SYNTHESIS EPISTEMIC MANDATE" in mandate


def test_thesis_balance_checker_flags_single_theme():
    from app.services.report_agent import ThesisBalanceChecker

    checker = ThesisBalanceChecker()
    draft = (
        "The merger announcement dominates outlook. The deal headline drives all narrative. "
        "Acquisition talks continue to anchor expectations."
    )
    report = checker.check(
        draft,
        "Stock market valuation impact",
        "Financial implications",
    )
    assert report.warning is not None
    assert "THESIS" in report.warning


def test_causal_completeness_flags_shallow_event_sentences():
    checker = CausalCompletenessChecker()

    draft = (
        "The merger was announced yesterday [S1]. "
        "The acquisition deal was reported widely [S1]. "
        "Valuation will rise."
    )
    report = checker.check(draft, "Market valuation", "Deal impact")
    assert report.warning is not None
    assert "causal" in report.warning.lower() or "mechanism" in report.warning.lower()

