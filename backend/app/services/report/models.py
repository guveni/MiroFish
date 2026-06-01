"""
Report model classes, enums, and data models.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class ReportStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


class FreshnessClass(str, Enum):
    """How time-sensitive a topic is."""
    HIGH = "high"      # markets, news, prices, politics, laws, product specs
    MEDIUM = "medium"  # company strategy, analyst views, regulations
    LOW = "low"        # historical / background facts


@dataclass
class TemporalAnnotation:
    dates_found: List[str]
    freshness_class: FreshnessClass
    freshness_score: float   # 0.0 (very stale) → 1.0 (fresh / undated)
    staleness_warning: Optional[str]
    header: str              # pre-formatted text to prepend to the tool result


@dataclass
class GroundingReport:
    """Summary of how well a generated section is anchored to retrieved sources."""
    total_sources: int
    cited_sources: List[int]         # [SN] numbers found in the draft
    uncited_sources: List[int]       # retrieved sources the LLM never cited
    grounding_score: float           # cited / total (0–1)
    high_risk_sentences: List[str]   # factual sentences without a nearby citation
    warning: Optional[str]
    summary: str


@dataclass
class CredibilityReport:
    """Credibility metadata score and cross-referenced claims report."""
    source_weights: Dict[int, str]          # e.g., {1: "HIGH", 2: "MEDIUM"}
    cross_referenced_claims: List[str]      # sentences citing >=2 sources
    summary: str


@dataclass
class ContradictionReport:
    """Internal contradictions detection report."""
    contradictions: List[Tuple[str, int, int, str]]  # (entity, S_a, S_b, reason)
    warning: Optional[str]
    summary: str
    detailed_contradictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NumericalSanityReport:
    """Numerical consistency and range-sanity report."""
    anomalies: List[str]
    warning: Optional[str]
    summary: str
    detailed_anomalies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SpecificityReport:
    """Vague filler and non-specific sentence detection report."""
    vague_sentences: List[str]
    warning: Optional[str]
    summary: str


@dataclass
class ConfidenceReport:
    """Confidence calibration and speculative claim alignment report."""
    uncalibrated_speculations: List[str]      # speculative claims lacking confidence label
    overconfident_unbacked_claims: List[str]  # confident claims without source
    warning: Optional[str]
    summary: str


@dataclass
class SkepticismReport:
    """Report on overstated certainty, evidence-confidence mismatches, and missing alternatives."""
    overstated_certainty: bool
    mismatches: List[str]
    missing_alternatives: bool
    warning: Optional[str]
    summary: str


@dataclass
class EvidenceAlignmentReport:
    """
    Heuristic claim-to-evidence alignment verifier.

    Goal: detect "analytical-looking" sentences whose cited claims don't
    actually appear in the cited evidence text (especially for numeric claims).
    """
    unsupported_sentences: List[str]
    alignment_score: float          # supported_claims / assessed_claims (0–1)
    warning: Optional[str]
    summary: str


@dataclass
class AnchoringDistributionReport:
    warning: Optional[str]
    summary: str


@dataclass
class CausalCompletenessReport:
    warning: Optional[str]
    missing_components: List[str]
    summary: str


@dataclass
class NumericalGroundingCoverageReport:
    warning: Optional[str]
    missing_components: List[str]
    summary: str


@dataclass
class ThesisBalanceReport:
    """Checks whether the draft balances multiple driver classes instead of one theme."""
    buckets_present: List[str]
    buckets_missing: List[str]
    dominant_bucket: Optional[str]
    warning: Optional[str]
    summary: str


@dataclass
class EpistemicReviewReport:
    """Post-generation epistemic critique: signal vs noise, primary vs secondary evidence."""
    evidence_quality_collapse: bool
    overconfident_low_tier_sentences: List[str]
    missing_primary_anchor: bool
    low_tier_citation_share: float
    critique_questions: List[str]
    warning: Optional[str]
    summary: str


@dataclass
class AnalyticalReport:
    """Aggregates all metric reports for analytical audit input to Composer."""
    grounding: GroundingReport
    credibility: CredibilityReport
    contradictions: ContradictionReport
    numerical_sanity: NumericalSanityReport
    specificity: SpecificityReport
    confidence: ConfidenceReport
    evidence_alignment: EvidenceAlignmentReport
    anchoring: AnchoringDistributionReport
    causal_completeness: CausalCompletenessReport
    numerical_grounding_coverage: NumericalGroundingCoverageReport
    thesis_balance: Optional[ThesisBalanceReport] = None
    skepticism: Optional[SkepticismReport] = None
    epistemic_review: Optional[EpistemicReviewReport] = None

    def to_composer_block(self) -> str:
        lines = ["[Analytical Audit Results — issues requiring immediate resolution]"]
        
        warnings = []
        if self.grounding.warning:
            warnings.append(self.grounding.warning)
        if self.contradictions.warning:
            warnings.append(self.contradictions.warning)
        if self.numerical_sanity.warning:
            warnings.append(self.numerical_sanity.warning)
        if self.specificity.warning:
            warnings.append(self.specificity.warning)
        if self.confidence.warning:
            warnings.append(self.confidence.warning)
        if self.evidence_alignment.warning:
            warnings.append(self.evidence_alignment.warning)
        if self.anchoring.warning:
            warnings.append(self.anchoring.warning)
        if self.causal_completeness.warning:
            warnings.append(self.causal_completeness.warning)
        if self.numerical_grounding_coverage.warning:
            warnings.append(self.numerical_grounding_coverage.warning)
        if self.skepticism and self.skepticism.warning:
            warnings.append(self.skepticism.warning)
        if self.thesis_balance and self.thesis_balance.warning:
            warnings.append(self.thesis_balance.warning)
        if self.epistemic_review and self.epistemic_review.warning:
            warnings.append(self.epistemic_review.warning)

        if warnings:
            lines.append("Audit Warnings:")
            for w in warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("No critical issues detected. Proceed with polishing.")

        if self.grounding.high_risk_sentences:
            lines.append("\nUnanchored Factual Sentences (cite sources inline using [S1] etc or reframe as inference):")
            for s in self.grounding.high_risk_sentences:
                lines.append(f"  - {s}")

        if self.contradictions.contradictions:
            lines.append("\nInternal Contradictions to resolve:")
            for ent, s1, s2, reason in self.contradictions.contradictions:
                lines.append(f"  - {reason}")

        if self.numerical_sanity.anomalies:
            lines.append("\nNumerical Sanity Anomalies to correct:")
            for anom in self.numerical_sanity.anomalies:
                lines.append(f"  - {anom}")

        if self.specificity.vague_sentences:
            lines.append("\nVague filler sentences to specify or remove:")
            for s in self.specificity.vague_sentences:
                lines.append(f"  - {s}")

        if self.confidence.uncalibrated_speculations:
            lines.append("\nSpeculative claims lacking confidence calibration labels:")
            for s in self.confidence.uncalibrated_speculations:
                lines.append(f"  - {s}")

        if self.confidence.overconfident_unbacked_claims:
            lines.append("\nOverconfident claims lacking factual sources:")
            for s in self.confidence.overconfident_unbacked_claims:
                lines.append(f"  - {s}")

        if self.skepticism and self.skepticism.mismatches:
            lines.append("\nEvidence-Confidence Mismatches to resolve (hedge or check sources):")
            for m in self.skepticism.mismatches:
                lines.append(f"  - {m}")

        if self.evidence_alignment.unsupported_sentences:
            lines.append("\nEvidence Alignment Failures (cited claim likely not present in cited evidence):")
            for s in self.evidence_alignment.unsupported_sentences:
                lines.append(f"  - {s}")

        if self.causal_completeness.missing_components:
            lines.append("\nCausal chain gaps to expand (event → mechanism → measurable → valuation):")
            for m in self.causal_completeness.missing_components:
                lines.append(f"  - {m}")

        if self.thesis_balance and self.thesis_balance.buckets_missing:
            lines.append(
                f"\nThesis balance — integrate missing driver lenses: {', '.join(self.thesis_balance.buckets_missing[:4])}"
            )

        if self.epistemic_review:
            if self.epistemic_review.overconfident_low_tier_sentences:
                lines.append(
                    "\nOverconfident claims on weak epistemic sources (hedge, downgrade, or re-cite):"
                )
                for s in self.epistemic_review.overconfident_low_tier_sentences:
                    lines.append(f"  - {s}")
            lines.append("\n[Epistemic critique — answer in revised prose]")
            for q in self.epistemic_review.critique_questions:
                lines.append(f"  - {q}")

        lines.append(f"\nSource Quality Map: {self.credibility.summary}")

        return "\n".join(lines)

    def severity_score(self) -> int:
        """Higher = more audit failures requiring aggressive rewrite."""
        score = 0
        if self.confidence.warning:
            score += 2
        if self.skepticism and self.skepticism.warning:
            score += 2
        if self.epistemic_review and self.epistemic_review.warning:
            score += 2
        if self.anchoring.warning:
            score += 1
        if self.evidence_alignment.warning:
            score += 1
        score += len(self.confidence.overconfident_unbacked_claims)
        score += len(self.confidence.uncalibrated_speculations)
        if self.skepticism:
            score += len(self.skepticism.mismatches)
        if self.epistemic_review:
            score += len(self.epistemic_review.overconfident_low_tier_sentences)
        return score

    def requires_confidence_rewrite(self, threshold: int = 3) -> bool:
        return self.severity_score() >= threshold

    def to_composer_block_with_evidence(
        self,
        evidence_scores: Optional[Dict[int, Any]] = None,
    ) -> str:
        block = self.to_composer_block()
        if not evidence_scores:
            return block
        scores = list(evidence_scores.values())
        if not scores:
            return block
        from ..evidence_evaluator import EvidenceEvaluator

        evaluator = EvidenceEvaluator()
        profile = evaluator.compute_section_profile(scores)
        extra = [
            "",
            evaluator.build_confidence_governed_block(profile),
            evaluator.build_narrative_weight_allocation(scores),
        ]
        mandate = evaluator.build_synthesis_mandate(scores)
        if mandate:
            extra.append(mandate)
        return block + "\n" + "\n".join(extra)
