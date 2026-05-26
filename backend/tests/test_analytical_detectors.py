import pytest
from app.services.report_agent import (
    SourceCredibilityScorer,
    ContradictionDetector,
    NumericalSanityChecker,
    SpecificityDetector,
    ConfidenceCoverageChecker,
    AnalyticalReport,
    GroundingReport,
    ReportAgent
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
