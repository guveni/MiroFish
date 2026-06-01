"""
Evaluation checkers and filters for Report Agent.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from .models import (
    FreshnessClass,
    TemporalAnnotation,
    GroundingReport,
    CredibilityReport,
    ContradictionReport,
    NumericalSanityReport,
    SpecificityReport,
    ConfidenceReport,
    SkepticismReport,
    EvidenceAlignmentReport,
    AnchoringDistributionReport,
    CausalCompletenessReport,
    NumericalGroundingCoverageReport,
    ThesisBalanceReport,
    EpistemicReviewReport,
)
from ..epistemic_discriminator import EpistemicDiscriminator


class TemporalRelevanceFilter:
    """
    Sits between tool retrieval and LLM synthesis.

    For each tool result it:
      1. Extracts all recognisable source/event dates.
      2. Classifies the topic's freshness requirement.
      3. Scores how fresh the retrieved content is.
      4. Prepends a compact Temporal Context header so the LLM can
         cite "as of [date]", flag stale sources, and resolve conflicts.
    """

    # Keywords that signal a high-freshness topic
    _HIGH_KW = {
        "market", "price", "stock", "bond", "crypto", "currency", "forex",
        "news", "breaking", "election", "vote", "politics", "political",
        "law", "legislation", "bill", "policy", "sanction", "tariff", "trade",
        "product", "launch", "release", "spec", "specification",
        "crisis", "conflict", "war", "pandemic", "outbreak", "disaster",
        "inflation", "interest rate", "gdp", "earnings", "quarterly", "ipo",
        "regulation", "fda", "sec", "ban",
    }

    # Keywords that signal a medium-freshness topic
    _MEDIUM_KW = {
        "strategy", "roadmap", "analyst", "forecast", "outlook",
        "corporate", "company", "merger", "acquisition", "partnership",
        "executive", "ceo", "leadership", "restructure",
        "survey", "report", "study", "trend", "sentiment",
    }

    # Compiled date patterns (ordered most-specific first)
    _RE_ISO_FULL    = re.compile(r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b')
    _RE_ISO_YM      = re.compile(r'\b(20\d{2})[-/](0[1-9]|1[0-2])\b')
    _RE_MONTH_YEAR  = re.compile(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(20\d{2})\b',
        re.IGNORECASE,
    )
    _RE_YEAR_ONLY   = re.compile(r'\b(20[12]\d)\b')

    # Map 3-letter month abbreviations to numbers
    _MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }

    # Staleness thresholds in days per freshness class
    _THRESHOLDS = {
        FreshnessClass.HIGH:   [(30, 0.9, None), (90, 0.6, "aging"), (365, 0.4, "outdated"), (None, 0.1, "stale")],
        FreshnessClass.MEDIUM: [(180, 0.9, None), (365, 0.6, "aging"), (730, 0.4, "outdated"), (None, 0.2, "stale")],
        FreshnessClass.LOW:    [(1825, 0.9, None), (3650, 0.6, "aging"), (None, 0.3, "old")],
    }

    # ── Public API ──────────────────────────────────────────────

    def classify_topic(self, text: str) -> FreshnessClass:
        """Return freshness class for the combined topic text."""
        lower = text.lower()
        if any(kw in lower for kw in self._HIGH_KW):
            return FreshnessClass.HIGH
        if any(kw in lower for kw in self._MEDIUM_KW):
            return FreshnessClass.MEDIUM
        return FreshnessClass.LOW

    def extract_dates(self, text: str) -> List[datetime]:
        """
        Extract all recognisable dates from text.
        Returns a list of datetime objects sorted newest-first.
        """
        found: set[datetime] = set()

        for m in self._RE_ISO_FULL.finditer(text):
            try:
                found.add(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass

        for m in self._RE_MONTH_YEAR.finditer(text):
            month_num = self._MONTH_MAP.get(m.group(1).lower()[:3])
            if month_num:
                try:
                    found.add(datetime(int(m.group(2)), month_num, 1))
                except ValueError:
                    pass

        for m in self._RE_ISO_YM.finditer(text):
            try:
                found.add(datetime(int(m.group(1)), int(m.group(2)), 1))
            except ValueError:
                pass

        # Year-only as a fallback when nothing more specific is found
        if not found:
            for m in self._RE_YEAR_ONLY.finditer(text):
                yr = int(m.group(1))
                if 2010 <= yr <= 2035:
                    try:
                        found.add(datetime(yr, 1, 1))
                    except ValueError:
                        pass

        return sorted(found, reverse=True)

    def score_freshness(
        self, dates: List[datetime], freshness_class: FreshnessClass
    ) -> Tuple[float, Optional[str]]:
        """
        Return (score 0..1, warning_message or None).
        Score is 0.7 (neutral) when no dates are found.
        """
        if not dates:
            return 0.7, None

        now = datetime.now()
        days_old = (now - dates[0]).days

        for max_days, score, label in self._THRESHOLDS[freshness_class]:
            if max_days is None or days_old <= max_days:
                warning = None
                if label:
                    approx = f"~{days_old // 365}y" if days_old >= 365 else f"~{days_old}d"
                    warning = f"⚠️ SOURCE {label.upper()} ({approx} old) — {freshness_class.value}-freshness topic; verify with current data."
                return score, warning

        # Should never reach here, but be safe
        return 0.1, "⚠️ SOURCE VERY STALE — treat with caution."

    def annotate(
        self,
        raw_result: str,
        simulation_requirement: str,
        section_title: str,
        tool_name: str,
    ) -> str:
        """
        Prepend a Temporal Context block to a raw tool result.
        The block tells the LLM how fresh the data is and what to do about it.
        """
        topic_text = f"{simulation_requirement} {section_title}"
        fc = self.classify_topic(topic_text)
        dates = self.extract_dates(raw_result)
        score, warning = self.score_freshness(dates, fc)

        lines = [f"[Temporal Context — {tool_name}]"]
        lines.append(f"Freshness requirement: {fc.value.upper()}  |  Source freshness score: {score:.1f}/1.0")

        if dates:
            date_strs = [d.strftime("%Y-%m-%d") for d in dates[:5]]
            lines.append(f"Dates found: {', '.join(date_strs)}")
            lines.append(f"Most recent: {dates[0].strftime('%Y-%m-%d')}")
        else:
            lines.append("Dates found: none detected in this source")

        if warning:
            lines.append(warning)

        lines += [
            "Synthesis rules for this source:",
            "  • Cite key claims with 'as of [date]' when a date is available.",
            "  • If this source conflicts with a newer source, prefer the newer one and note the discrepancy.",
            "  • If the source is stale for a high-freshness topic, flag this limitation in your analysis.",
            "---",
        ]

        header = "\n".join(lines)
        return f"{header}\n\n{raw_result}"

    def build_annotation(
        self,
        raw_result: str,
        simulation_requirement: str,
        section_title: str,
        tool_name: str,
    ) -> TemporalAnnotation:
        """Return a structured TemporalAnnotation (useful for logging)."""
        topic_text = f"{simulation_requirement} {section_title}"
        fc = self.classify_topic(topic_text)
        dates = self.extract_dates(raw_result)
        score, warning = self.score_freshness(dates, fc)
        date_strs = [d.strftime("%Y-%m-%d") for d in dates[:5]]
        return TemporalAnnotation(
            dates_found=date_strs,
            freshness_class=fc,
            freshness_score=score,
            staleness_warning=warning,
            header=self.annotate(raw_result, simulation_requirement, section_title, tool_name),
        )


class GroundingVerifier:
    """
    Heuristic grounding checker — no extra LLM call.

    After the LLM writes a section, it checks:
    1. Which numbered sources ([S1] … [SN]) were actually cited.
    2. Which sentences look factual but carry no citation.
    3. Produces a score and optional warning that travels to the composer.
    """

    # Citation pattern: [S1], [S2], [S 3], [s4] …
    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    # Sentence-level indicators of a specific factual claim
    _FACTUAL_RE = re.compile(
        r'\d[\d,]*\.?\d*\s*%|\b20\d{2}\b|"\w|\b(?:said|stated|reported|claimed|announced|according to)\b|\b(?:increased|decreased|grew|declined|surged|plunged|rose|fell)\b|\$[\d,]+|\b\d+\s+(?:million|billion|thousand)\b',
        re.IGNORECASE,
    )

    def check(self, draft: str, total_sources: int) -> GroundingReport:
        """Analyse citation coverage of *draft* against *total_sources* numbered sources."""
        if total_sources == 0:
            return GroundingReport(
                total_sources=0,
                cited_sources=[],
                uncited_sources=[],
                grounding_score=1.0,
                high_risk_sentences=[],
                warning=None,
                summary="No sources retrieved for this section.",
            )

        cited_nums = {int(m.group(1)) for m in self._CITATION_RE.finditer(draft)}
        all_nums = set(range(1, total_sources + 1))
        cited = sorted(cited_nums & all_nums)
        uncited = sorted(all_nums - cited_nums)
        score = len(cited) / total_sources

        # Collect sentences that carry a factual signal but no [SN] tag
        high_risk: List[str] = []
        for sent in re.split(r'(?<=[.!?])\s+', draft):
            sent = sent.strip()
            if (
                len(sent) > 30
                and self._FACTUAL_RE.search(sent)
                and not self._CITATION_RE.search(sent)
            ):
                high_risk.append(sent[:150] + ("…" if len(sent) > 150 else ""))
                if len(high_risk) >= 5:
                    break

        if score < 0.5:
            warning = f"⚠️ LOW GROUNDING: only {len(cited)}/{total_sources} sources cited. Verify claims against observations."
        elif uncited:
            warning = f"ℹ️ {len(uncited)} source(s) unused: {', '.join(f'S{n}' for n in uncited)}. Consider whether they contain relevant evidence."
        else:
            warning = None

        parts = [f"Grounding {score:.0%} ({len(cited)}/{total_sources} sources cited)"]
        if uncited:
            parts.append(f"Uncited: {', '.join(f'S{n}' for n in uncited)}")
        if high_risk:
            parts.append(f"Unanchored factual sentences: {len(high_risk)}")

        return GroundingReport(
            total_sources=total_sources,
            cited_sources=cited,
            uncited_sources=uncited,
            grounding_score=score,
            high_risk_sentences=high_risk,
            warning=warning,
            summary=" | ".join(parts),
        )


class SourceCredibilityScorer:
    """
    Assigns epistemic weight to sources based on the tool that retrieved them,
    and identifies cross-referenced claims in the draft.
    """
    _TOOL_WEIGHTS = {
        "interview_agents": "HIGH",
        "insight_forge": "HIGH",
        "panorama_search": "MEDIUM",
        "quick_search": "MEDIUM",
        "web_search": "LOW"
    }

    def check(self, draft: str, sources_metadata: List[Tuple[int, str, str]]) -> CredibilityReport:
        source_weights = {}
        for num, tool_name, _ in sources_metadata:
            source_weights[num] = self._TOOL_WEIGHTS.get(tool_name, "LOW")

        # Find cross-referenced claims
        cross_referenced = []
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        
        citation_re = re.compile(r'\[[Ss]\s*(\d+)\]')
        for sent in re.split(r'(?<=[.!?])\s+', draft_clean):
            sent = sent.strip()
            if len(sent) > 20:
                cited_nums = {int(m.group(1)) for m in citation_re.finditer(sent)}
                if len(cited_nums) >= 2:
                    cross_referenced.append(sent[:150] + ("…" if len(sent) > 150 else ""))

        parts = [f"Credibility breakdown: {', '.join(f'S{k}:{v}' for k, v in source_weights.items())}"]
        if cross_referenced:
            parts.append(f"Cross-referenced sentences: {len(cross_referenced)}")
        else:
            parts.append("No cross-referenced sentences found.")

        return CredibilityReport(
            source_weights=source_weights,
            cross_referenced_claims=cross_referenced,
            summary=" | ".join(parts)
        )


class ContradictionDetector:
    """
    Heuristically checks if sentences in the draft have opposing polarity
    regarding the same entity/concept.
    """
    _POS_WORDS = {
        "rose", "grew", "increased", "upward", "up", "support", "supportive", 
        "pro", "positive", "increase", "gain", "bullish", "optimistic", "gains",
        "benefits", "benefit", "boost", "boosted", "improved", "improvement", "success"
    }
    _NEG_WORDS = {
        "fell", "dropped", "declined", "downward", "down", "oppose", "opposing", 
        "con", "negative", "decrease", "loss", "bearish", "pessimistic", "plunged", 
        "slumped", "failures", "failure", "damage", "damaged", "hurt", "harmed", "losses"
    }

    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(self, draft: str) -> ContradictionReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        stop_words = {
            "The", "A", "An", "In", "On", "At", "By", "For", "With", "About", "Against", "Through", 
            "During", "Before", "After", "Under", "Above", "Below", "And", "Or", "But", "If", "Because", 
            "As", "Until", "While", "Of", "To", "It", "Its", "They", "Their", "We", "Our", "He", "She", 
            "This", "That", "These", "Those", "S1", "S2", "S3", "S4", "S5", "Final", "Answer", "Note", "Yes", "No",
            "Based", "According", "Agent", "Agents"
        }

        entity_data = []  # list of dicts: {"entity": str, "sent": str, "sources": set, "polarity": int}

        for sent in sentences:
            sources = {int(m.group(1)) for m in self._CITATION_RE.finditer(sent)}
            if not sources:
                continue

            words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', sent)
            entities = set(w for w in words if w not in stop_words and len(w) > 2)

            sent_lower = sent.lower()
            pos_matches = sum(1 for w in self._POS_WORDS if f" {w} " in f" {sent_lower} " or sent_lower.endswith(w) or sent_lower.startswith(w))
            neg_matches = sum(1 for w in self._NEG_WORDS if f" {w} " in f" {sent_lower} " or sent_lower.endswith(w) or sent_lower.startswith(w))

            polarity = 0
            if pos_matches > neg_matches:
                polarity = 1
            elif neg_matches > pos_matches:
                polarity = -1

            if polarity != 0:
                for ent in entities:
                    entity_data.append({
                        "entity": ent,
                        "sent": sent,
                        "sources": sources,
                        "polarity": polarity
                    })

        contradictions = []
        detailed_contradictions = []
        for i in range(len(entity_data)):
            for j in range(i + 1, len(entity_data)):
                d1 = entity_data[i]
                d2 = entity_data[j]
                if d1["entity"] == d2["entity"] and d1["polarity"] != d2["polarity"]:
                    src_1 = min(d1["sources"])
                    src_2 = min(d2["sources"])
                    if src_1 != src_2:
                        reason = f"Entity '{d1['entity']}' has positive/growth indicators in Sentence A ({src_1}) but negative/decline indicators in Sentence B ({src_2})."
                        contradictions.append((d1["entity"], src_1, src_2, reason))
                        detailed_contradictions.append({
                            "entity": d1["entity"],
                            "src_1": src_1,
                            "src_2": src_2,
                            "reason": reason,
                            "sent_1": d1["sent"],
                            "sent_2": d2["sent"]
                        })

        if contradictions:
            warning = f"⚠️ CONTRADICTION DETECTED: Found opposing claims for: {', '.join(set(c[0] for c in contradictions))}. Resolve conflicting evidence or clarify temporal order."
            summary = f"Contradictions found: {len(contradictions)}"
        else:
            warning = None
            summary = "No contradictions detected."

        return ContradictionReport(
            contradictions=contradictions,
            warning=warning,
            summary=summary,
            detailed_contradictions=detailed_contradictions
        )


class NumericalSanityChecker:
    """
    Checks numbers/percentages in the draft for logical consistency, sum rules,
    and order-of-magnitude alignment with the cited sources.
    """
    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(self, draft: str, sources_metadata: List[Tuple[int, str, str]], module: Optional[Any] = None) -> NumericalSanityReport:
        anomalies = []
        detailed_anomalies = []
        if module:
            # Consume domain-specific sanity checks
            for domain_anom in module.check_numerical_sanity(draft, sources_metadata):
                anomalies.append(domain_anom)
                detailed_anomalies.append({
                    "type": "domain_sanity",
                    "sentence": "",
                    "message": domain_anom
                })
            if hasattr(module, 'check_quantitative_grounding'):
                for domain_ground in module.check_quantitative_grounding(draft, sources_metadata):
                    anomalies.append(domain_ground)
                    detailed_anomalies.append({
                        "type": "domain_grounding",
                        "sentence": "",
                        "message": domain_ground
                    })
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        # 1. Percentage Range and Sum Sanity
        for sent in sentences:
            pct_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*%', sent)
            if not pct_matches:
                continue

            pct_vals = [float(v) for v in pct_matches]
            sent_lower = sent.lower()

            # Individual percentage > 100% check when describing support/proportion/share
            if any(term in sent_lower for term in ["support", "sentiment", "share", "proportion", "percentage", "rate"]):
                for val in pct_vals:
                    if val > 100.0:
                        anomaly_msg = f"Individual percentage {val}% exceeds 100% in a proportional context: '{sent[:80]}...'"
                        anomalies.append(anomaly_msg)
                        detailed_anomalies.append({
                            "type": "proportional_exceed",
                            "sentence": sent,
                            "message": anomaly_msg
                        })

            # Check sums if there are multiple percentages in the sentence
            if len(pct_vals) >= 2 and any(term in sent_lower for term in ["total", "sum", "combine", "support", "oppose", "neutral"]):
                total_pct = sum(pct_vals)
                if any(term in sent_lower for term in ["breakdown", "split", "distribute", "divided"]) or (
                    any(p in sent_lower for p in ["support", "oppose"]) and any(n in sent_lower for n in ["neutral", "observer"])
                ):
                    if total_pct > 105.0 or total_pct < 90.0:
                        anomaly_msg = f"Breakdown percentages sum to {total_pct}% (should be ~100%): '{sent[:80]}...'"
                        anomalies.append(anomaly_msg)
                        detailed_anomalies.append({
                            "type": "breakdown_sum",
                            "sentence": sent,
                            "message": anomaly_msg
                        })

        # 2. Order of Magnitude Mismatch with cited observation
        sources_dict = {num: (tool, result) for num, tool, result in sources_metadata}

        for sent in sentences:
            cited_nums = {int(m.group(1)) for m in self._CITATION_RE.finditer(sent)}
            if not cited_nums:
                continue

            num_matches = re.finditer(r'\b(\d+(?:\.\d+)?)\s*(%|\bmillion\b|\bbillion\b|\bthousand\b)?', sent, re.IGNORECASE)
            for nm in num_matches:
                val_str = nm.group(1)
                suffix = nm.group(2) or ""
                suffix = suffix.lower().strip()
                val = float(val_str)

                if val < 5.0 and not suffix:
                    continue

                for src_num in cited_nums:
                    if src_num not in sources_dict:
                        continue
                    _, src_text = sources_dict[src_num]

                    if suffix in ["million", "billion", "thousand"]:
                        src_matches = re.finditer(r'\b' + re.escape(val_str) + r'\b\s*(%|\bmillion\b|\bbillion\b|\bthousand\b)?', src_text, re.IGNORECASE)
                        matched_in_src = False
                        different_magnitude = False
                        for sm in src_matches:
                            matched_in_src = True
                            src_suffix = sm.group(1) or ""
                            src_suffix = src_suffix.lower().strip()
                            if src_suffix != suffix:
                                different_magnitude = True
                                break
                        if matched_in_src and different_magnitude:
                            anomaly_msg = f"Potential order of magnitude mismatch for {val_str} ({suffix} in draft vs different scale in source [S{src_num}])."
                            anomalies.append(anomaly_msg)
                            detailed_anomalies.append({
                                "type": "order_of_magnitude",
                                "sentence": sent,
                                "message": anomaly_msg,
                                "source_num": src_num
                            })

        if anomalies:
            warning = f"⚠️ NUMERICAL SANITY WARNING: Found {len(anomalies)} mathematical or scale anomalies. Double-check percentage breakdowns and scaling."
            summary = f"Numerical anomalies: {len(anomalies)}"
        else:
            warning = None
            summary = "Numerical sanity verified."

        return NumericalSanityReport(
            anomalies=anomalies,
            warning=warning,
            summary=summary,
            detailed_anomalies=detailed_anomalies
        )


class SpecificityDetector:
    """
    Scans the draft for generic or vague filler phrases that lack specificity,
    flagging them if they are not backed by sources or are outside the self-critique block.
    """
    _VAGUE_PHRASES = [
        "monitor closely", "potential opportunities", "careful management", "navigate uncertainty",
        "could pose risks", "may impact", "in the long run", "going forward", "closely monitor",
        "potential risk", "careful oversight", "strategic implications", "potential for",
        "should be watched", "uncertain future", "requires attention"
    ]

    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(self, draft: str, module: Optional[Any] = None) -> SpecificityReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        vague_phrases = self._VAGUE_PHRASES[:]
        if module:
            vague_phrases.extend(module.get_vague_phrases())

        vague_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(phrase in sent_lower for phrase in vague_phrases):
                if not self._CITATION_RE.search(sent):
                    vague_sentences.append(sent[:150] + ("…" if len(sent) > 150 else ""))

        if vague_sentences:
            warning = f"⚠️ VAGUENESS DETECTED: Found {len(vague_sentences)} sentences using generic LLM filler language without source backing. Replace with specific evidence or delete."
            summary = f"Vague sentences: {len(vague_sentences)}"
        else:
            warning = None
            summary = "No unbacked vagueness detected."

        return SpecificityReport(
            vague_sentences=vague_sentences,
            warning=warning,
            summary=summary
        )


class ConfidenceCoverageChecker:
    """
    Verifies that speculative claims carry appropriate confidence labels,
    and flags overly confident assertions that lack inline sources.
    """
    _SPECULATIVE_KEYWORDS = ["infer", "suggests", "could", "may", "might", "speculate", "perhaps", "possibly", "potential", "scenario"]
    _CONFIDENCE_LABELS = [
        "high confidence", "medium confidence", "low confidence", "moderately likely", 
        "highly likely", "highly speculative", "probability", "probabilistic", "confidence level",
        "certainty", "uncertainty", "speculatively", "inferred", "estimated", "modeled", "measured"
    ]
    _CONFIDENT_KEYWORDS = ["will", "definitely", "clearly", "obviously", "certainly", "proves", "undoubtedly", "always"]

    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(self, draft: str) -> ConfidenceReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        uncalibrated_speculations = []
        overconfident_unbacked_claims = []

        for sent in sentences:
            sent_lower = sent.lower()
            has_source = bool(self._CITATION_RE.search(sent))

            is_speculative = any(w in sent_lower for w in self._SPECULATIVE_KEYWORDS)
            if is_speculative:
                has_label = any(lbl in sent_lower for lbl in self._CONFIDENCE_LABELS)
                if not has_label:
                    uncalibrated_speculations.append(sent[:150] + ("…" if len(sent) > 150 else ""))

            is_confident = any(w in sent_lower for w in self._CONFIDENT_KEYWORDS)
            if is_confident and not has_source:
                overconfident_unbacked_claims.append(sent[:150] + ("…" if len(sent) > 150 else ""))

        warnings = []
        if uncalibrated_speculations:
            warnings.append(f"{len(uncalibrated_speculations)} speculative claims lack confidence labels")
        if overconfident_unbacked_claims:
            warnings.append(f"{len(overconfident_unbacked_claims)} confident claims lack source citations")

        if warnings:
            warning = f"⚠️ CONFIDENCE CALIBRATION ISSUES: {'; '.join(warnings)}. Calibrate your assertions."
            summary = f"Uncalibrated spec: {len(uncalibrated_speculations)} | Overconfident: {len(overconfident_unbacked_claims)}"
        else:
            warning = None
            summary = "Confidence calibration verified."

        return ConfidenceReport(
            uncalibrated_speculations=uncalibrated_speculations,
            overconfident_unbacked_claims=overconfident_unbacked_claims,
            warning=warning,
            summary=summary
        )


class SkepticismChecker:
    """
    Evaluates draft for excessive confidence, unhedged claims backed by low-credibility
    or low-specificity evidence, and a lack of alternative/counter-scenario framing.
    """
    _CONFIDENT_KEYWORDS = ["will", "definitely", "clearly", "obviously", "certainly", "proves", "undoubtedly", "always"]
    _SPECULATIVE_KEYWORDS = ["could", "may", "might", "perhaps", "possibly", "potential", "suggests", "infer", "speculate"]
    _ALTERNATIVE_KEYWORDS = ["alternatively", "however", "counter-argument", "on the other hand", "conversely", "another possibility", "different perspective"]
    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(self, draft: str, evidence_scores: Dict[int, Any]) -> SkepticismReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        confident_count = 0
        speculative_count = 0
        has_alternatives = False

        mismatches = []

        for sent in sentences:
            sent_lower = sent.lower()
            
            # 1. Count confident vs speculative
            is_confident = any(w in sent_lower for w in self._CONFIDENT_KEYWORDS)
            is_speculative = any(w in sent_lower for w in self._SPECULATIVE_KEYWORDS)
            
            if is_confident:
                confident_count += 1
            if is_speculative:
                speculative_count += 1
                
            # 2. Check alternatives
            if any(w in sent_lower for w in self._ALTERNATIVE_KEYWORDS):
                has_alternatives = True
                
            # 3. Mismatch checks
            if is_confident:
                cited_nums = {int(m.group(1)) for m in self._CITATION_RE.finditer(sent)}
                for num in cited_nums:
                    if num in evidence_scores:
                        score = evidence_scores[num]
                        # If a definitive word is used, but credibility or specificity is low
                        tier = getattr(score, "epistemic_tier", "")
                        low_epistemic = tier in (
                            "RETAIL_COMMENTARY", "SOCIAL_OPINION", "SPECULATIVE", "ANALYST_ACTION",
                        )
                        weak_weight = getattr(score, "synthesis_weight", 1.0) < 0.45
                        if (
                            score.credibility < 0.5
                            or score.specificity < 0.4
                            or low_epistemic
                            or weak_weight
                        ):
                            mismatches.append(
                                f"Sentence uses definitive language '{sent[:50]}...' but cites "
                                f"[S{num}] tier={tier} (weight={getattr(score, 'synthesis_weight', 0):.2f}, "
                                f"credibility={score.credibility:.1f})"
                            )

        overstated_certainty = False
        ratio = 0.0
        if confident_count > 0:
            ratio = confident_count / (confident_count + speculative_count)
            if ratio > 0.60:
                overstated_certainty = True

        missing_alternatives = not has_alternatives

        warnings = []
        if overstated_certainty:
            warnings.append(f"Overstated certainty: {confident_count} confident sentences vs {speculative_count} speculative/hedged sentences (ratio {ratio:.1%}; recommend more hedging)")
        if mismatches:
            warnings.append(f"Evidence-confidence mismatches: {len(mismatches)} confident claims rely on low-quality evidence")
        if missing_alternatives:
            warnings.append("Missing alternative scenario or counter-argument framing (recommend incorporating other perspectives)")

        if warnings:
            warning = "⚠️ SKEPTICISM WARNING: " + "; ".join(warnings)
            summary = f"Overstated certainty: {overstated_certainty} | Mismatches: {len(mismatches)} | Missing alternatives: {missing_alternatives}"
        else:
            warning = None
            summary = "Skepticism and confidence balance verified."

        return SkepticismReport(
            overstated_certainty=overstated_certainty,
            mismatches=mismatches,
            missing_alternatives=missing_alternatives,
            warning=warning,
            summary=summary
        )


class EvidenceAlignmentChecker:
    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')
    _NUM_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
    _PCT_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*%')
    _DOLLAR_RE = re.compile(
        r'\$\s*(\d+(?:\.\d+)?)(?:\s*(million|billion|thousand))?\b', re.IGNORECASE
    )
    _SUFFIX_AMOUNT_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*(million|billion|thousand)\b', re.IGNORECASE)
    _DATE_RE = re.compile(r'\b20[1-3]\d\b')
    _ENTITY_RE = re.compile(r'\b[A-Z][a-zA-Z0-9_]+\b')

    _FINANCIAL_KEY_TERMS = {
        "revenue", "profit", "loss", "earnings", "margin", "ebitda",
        "market cap", "valuation", "p/e", "p/e ratio", "ev/ebitda",
        "multiple", "pe", "ev", "cost", "pricing", "tariff",
        "acquisition", "funding", "dilution", "accretion",
    }
    _MEASURABLE_SIGNAL_TERMS = {"revenue", "profit", "loss", "margin", "market cap", "valuation", "cost", "earnings"}

    def _extract_numeric_signatures(self, text: str) -> Dict[str, List[float]]:
        pct_vals = [float(v) for v in self._PCT_RE.findall(text)]
        dollar_vals: List[float] = []
        for v, _suffix in self._DOLLAR_RE.findall(text):
            dollar_vals.append(float(v))
        suffix_vals = [float(v) for v, _unit in self._SUFFIX_AMOUNT_RE.findall(text)]
        date_vals = [float(v) for v in self._DATE_RE.findall(text)]

        return {
            "pct": pct_vals,
            "amount": dollar_vals + suffix_vals,
            "date": date_vals,
        }

    @staticmethod
    def _floats_match(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    def _sentence_has_claim_signal(self, sentence_lower: str, numeric_sigs: Dict[str, List[float]]) -> bool:
        if numeric_sigs["pct"] or numeric_sigs["amount"] or numeric_sigs["date"]:
            return True
        return any(term in sentence_lower for term in self._FINANCIAL_KEY_TERMS)

    def _is_sentence_likely_quantified(self, sentence_lower: str, numeric_sigs: Dict[str, List[float]]) -> bool:
        return bool(numeric_sigs["pct"] or numeric_sigs["amount"])

    def check(
        self,
        draft: str,
        sources_metadata: List[Tuple[int, str, str]],
        evidence_scores: Dict[int, Any],
        min_alignment: float = 0.7,
    ) -> EvidenceAlignmentReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        sources_by_num = {num: result for num, _, result in sources_metadata}

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_clean) if len(s.strip()) > 15]

        assessed = 0
        supported = 0
        unsupported: List[str] = []

        for sent in sentences:
            cited_nums = {int(m.group(1)) for m in self._CITATION_RE.finditer(sent)}
            if not cited_nums:
                continue

            sent_lower = sent.lower()
            numeric_sigs = self._extract_numeric_signatures(sent)
            if not self._sentence_has_claim_signal(sent_lower, numeric_sigs):
                continue

            assessed += 1

            sent_entities = {e for e in self._ENTITY_RE.findall(sent) if len(e) > 3}
            sent_has_measurable_terms = any(t in sent_lower for t in self._MEASURABLE_SIGNAL_TERMS)
            sent_requires_numeric = self._is_sentence_likely_quantified(sent_lower, numeric_sigs)

            best_support = 0.0
            for num in cited_nums:
                src_text = sources_by_num.get(num, "")
                src_lower = src_text.lower()

                src_numeric_sigs = self._extract_numeric_signatures(src_text)

                numeric_support = 0.0
                if sent_requires_numeric:
                    pct_ok = any(self._floats_match(v, src_v) for v in numeric_sigs["pct"] for src_v in src_numeric_sigs["pct"])
                    amt_ok = any(self._floats_match(v, src_v) for v in numeric_sigs["amount"] for src_v in src_numeric_sigs["amount"])
                    numeric_support = 1.0 if (pct_ok or amt_ok) else 0.0
                else:
                    date_ok = any(
                        self._floats_match(v, src_v)
                        for v in numeric_sigs["date"]
                        for src_v in src_numeric_sigs["date"]
                    )
                    numeric_support = 1.0 if date_ok else 0.0

                entity_support = 0.0
                if sent_entities:
                    hits = sum(1 for ent in sent_entities if ent.lower() in src_lower)
                    entity_support = min(1.0, hits / max(1, len(sent_entities)))

                keyword_support = 0.0
                if any(term in sent_lower for term in self._FINANCIAL_KEY_TERMS) or sent_has_measurable_terms:
                    denom = max(1, len([t for t in self._FINANCIAL_KEY_TERMS if t in sent_lower]))
                    key_hits = sum(1 for term in self._FINANCIAL_KEY_TERMS if term in sent_lower and term in src_lower)
                    keyword_support = min(1.0, key_hits / denom)

                support = 0.6 * numeric_support + 0.25 * entity_support + 0.15 * keyword_support
                best_support = max(best_support, support)

            if best_support >= min_alignment:
                supported += 1
            else:
                unsupported.append(sent[:200] + ("…" if len(sent) > 200 else ""))

        if assessed == 0:
            return EvidenceAlignmentReport(
                unsupported_sentences=[],
                alignment_score=1.0,
                warning=None,
                summary="Evidence alignment not assessed (no extractable claim sentences with citations).",
            )

        alignment_score = supported / assessed
        if unsupported:
            warning = (
                f"⚠️ EVIDENCE ALIGNMENT WARNING: {len(unsupported)} cited claim(s) appear weakly supported by their cited sources. "
                "Remove/replace or re-cite stronger evidence."
            )
        else:
            warning = None

        summary = f"Evidence alignment: {supported}/{assessed} supported claim sentences ({alignment_score:.0%})."
        return EvidenceAlignmentReport(
            unsupported_sentences=unsupported[:5],
            alignment_score=alignment_score,
            warning=warning,
            summary=summary,
        )


class AnchoringDistributionChecker:
    """
    Detects narrative anchoring risk: one retrieved source dominates citations in the draft.
    """
    _CITATION_RE = re.compile(r'\[[Ss]\s*(\d+)\]')

    def check(
        self,
        draft: str,
        evidence_scores: Dict[int, Any],
        min_unique_sources: int = 3,
        top_share_threshold: float = 0.55,
    ) -> AnchoringDistributionReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)

        citations = [int(m.group(1)) for m in self._CITATION_RE.finditer(draft_clean)]
        if len(citations) < 3:
            return AnchoringDistributionReport(
                warning=None,
                summary="Anchoring distribution not assessed (few citations).",
            )

        counts: Dict[int, int] = {}
        for c in citations:
            counts[c] = counts.get(c, 0) + 1

        unique_sources = list(counts.keys())
        if len(unique_sources) < min_unique_sources:
            return AnchoringDistributionReport(
                warning=None,
                summary=f"Anchoring distribution not assessed (only {len(unique_sources)} unique cited sources).",
            )

        total = sum(counts.values())
        top_source = max(counts.items(), key=lambda kv: kv[1])[0]
        top_share = counts[top_source] / total

        if top_share <= top_share_threshold:
            return AnchoringDistributionReport(
                warning=None,
                summary=f"Anchoring distribution OK (top source S{top_source} share {top_share:.0%}).",
            )

        top_score = evidence_scores.get(top_source)
        if top_score:
            risk_str = f"(top source credibility={top_score.credibility:.1f}, specificity={top_score.specificity:.1f})"
        else:
            risk_str = "(no evidence score for top source)"

        warning = (
            f"⚠️ NARRATIVE ANCHORING RISK: Source [S{top_source}] accounts for {top_share:.0%} of all cited evidence. "
            f"This can overfit the narrative to one event. Re-balance by citing and weighting multiple mechanisms/timelines. "
            f"{risk_str}"
        )

        return AnchoringDistributionReport(
            warning=warning,
            summary=f"Anchoring risk detected (top source share {top_share:.0%}).",
        )


class CausalCompletenessChecker:
    _MECHANISM_CONNECTORS = {
        "due to", "driven by", "because", "as a result", "therefore",
        "through", "leading to", "results in", "caused by", "contributes to",
        "via", "by reducing", "by increasing",
    }
    _EVENT_SIGNALS = {
        "merger", "acquisition", "acquire", "deal", "partnership", "ipo",
        "announced", "reported", "layoff", "recall", "sanction", "ban",
        "lawsuit", "settlement", "strike", "outage", "breach",
    }
    _MARKET_MEASURABLE = {"revenue", "profit", "loss", "margin", "earnings", "cost", "market cap", "price", "demand", "supply", "ebitda"}
    _MARKET_VALUATION = {"valuation", "p/e", "pe", "ev", "dcf", "multiple", "ev/ebitda"}
    _DILUTION_ACCRETION = {"dilution", "accretion"}
    _ACQUISITION_FUNDING = {"acquisition", "funding", "raise", "funding round", "shares", "cap table"}

    _RISK_MEASURABLE = {"severity", "failure rate", "incident", "downtime", "risk score", "penalty", "compliance cost", "breach"}
    _RISK_STRATEGIC = {"strategy", "reputation", "market share", "competitive", "risk appetite", "regulatory"}

    def _count_shallow_event_claims(self, draft_clean: str) -> int:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", draft_clean)
            if len(s.strip()) > 20
        ]
        shallow = 0
        for sent in sentences:
            sent_lower = sent.lower()
            if not any(ev in sent_lower for ev in self._EVENT_SIGNALS):
                continue
            has_mechanism = any(conn in sent_lower for conn in self._MECHANISM_CONNECTORS)
            has_number = bool(re.search(r"\b\d+(?:\.\d+)?%|\$\s*\d", sent_lower))
            if not has_mechanism and not has_number:
                shallow += 1
        return shallow

    def check(self, draft: str, simulation_requirement: str, section_title: str) -> CausalCompletenessReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        text_lower = (simulation_requirement + " " + section_title + " " + draft_clean).lower()

        is_markets = any(k in text_lower for k in ["market", "valuation", "revenue", "profit", "earnings", "price", "stock"])

        mechanism_present = any(conn in text_lower for conn in self._MECHANISM_CONNECTORS)
        missing: List[str] = []
        if not mechanism_present:
            missing.append("mechanism (how/why the event changes outcomes)")

        shallow_events = self._count_shallow_event_claims(draft_clean)
        if shallow_events >= 2:
            missing.append(
                f"deep causal links ({shallow_events} event-only sentences lack mechanism→measurable impact)"
            )
        elif shallow_events == 1 and not mechanism_present:
            missing.append("deep causal link for the primary event claim (mechanism + measurable impact)")

        if is_markets:
            measurable_present = any(k in text_lower for k in self._MARKET_MEASURABLE) or bool(
                re.search(r'\b\d+(?:\.\d+)?\s*%|\$\s*\d', text_lower)
            )
            if not measurable_present:
                missing.append("measurable impact (revenue/cost/margins/price, with numbers where possible)")

            valuation_present = any(k in text_lower for k in self._MARKET_VALUATION)
            if not valuation_present:
                missing.append("valuation consequence (valuation metric or multiple)")

            needs_dilution = any(k in text_lower for k in self._ACQUISITION_FUNDING)
            if needs_dilution and not any(k in text_lower for k in self._DILUTION_ACCRETION):
                missing.append("dilution/accretion math (if acquisition/funding is discussed)")
        else:
            measurable_present = any(k in text_lower for k in self._RISK_MEASURABLE)
            if not measurable_present:
                missing.append("measurable operational impact (risk score, severity, downtime, penalties)")

            strategic_present = any(k in text_lower for k in self._RISK_STRATEGIC)
            if not strategic_present:
                missing.append("strategic outcome (reputation/compliance/market share implications)")

        if missing:
            warning = (
                "⚠️ CAUSAL CHAIN INCOMPLETE: Missing "
                + "; ".join(missing)
                + ". Expand as: event -> mechanism -> measurable impact -> valuation/strategic consequence."
            )
        else:
            warning = None

        summary = f"Causal chain coverage: {('OK' if not missing else 'Missing ' + str(len(missing)) + ' component(s)')}."
        return CausalCompletenessReport(
            warning=warning,
            missing_components=missing[:5],
            summary=summary,
        )


class NumericalGroundingCoverageChecker:
    _SCENARIO_WORDS = {"scenario", "base case", "upside", "downside", "probability-weighted", "weighted outcome"}
    _PROBABILITY_WORDS = {"probability", "probabilistic", "weighted", "probability-weighted", "% probability"}
    _VALUATION_WORDS = {"valuation", "p/e", "pe", "ev", "dcf", "multiple", "ev/ebitda"}
    _SENSITIVITY_WORDS = {"sensitivity", "what-if", "range analysis", "sensitivity analysis"}
    _DILUTION_WORDS = {"dilution", "accretion"}
    _ACQUISITION_FUNDING_WORDS = {"acquisition", "funding", "raise", "cap table", "shares"}

    _HAS_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

    def check(self, draft: str, simulation_requirement: str, section_title: str) -> NumericalGroundingCoverageReport:
        draft_clean = re.sub(r'<self_critique>.*?</self_critique>', '', draft, flags=re.DOTALL)
        text_lower = (simulation_requirement + " " + section_title + " " + draft_clean).lower()

        has_any_numbers = bool(self._HAS_NUMBER_RE.search(text_lower))
        has_scenario = any(w in text_lower for w in self._SCENARIO_WORDS)
        has_probability = any(w in text_lower for w in self._PROBABILITY_WORDS)
        has_valuation = any(w in text_lower for w in self._VALUATION_WORDS)
        has_sensitivity = any(w in text_lower for w in self._SENSITIVITY_WORDS)
        has_dilution = any(w in text_lower for w in self._DILUTION_WORDS)
        has_acquisition_funding = any(w in text_lower for w in self._ACQUISITION_FUNDING_WORDS)

        missing: List[str] = []

        if has_valuation and not has_any_numbers:
            missing.append("valuation consequence quantified with numbers ($/multiples/%, etc.)")

        if (has_scenario or has_valuation) and not has_probability:
            missing.append("probability-weighted outcomes (scenario probabilities or % probability labels)")

        if has_acquisition_funding and not has_dilution:
            missing.append("dilution/accretion math or cap-table impact terms")

        # Flag sensitivity only when scenario/range language is already present.
        if (has_scenario or "range" in text_lower) and not has_sensitivity:
            missing.append("sensitivity analysis (what changes under alternative assumptions)")

        if missing:
            warning = (
                "⚠️ NUMERICAL GROUNDING INCOMPLETE: Missing "
                + "; ".join(missing)
                + ". Add probability-weighted outcomes and at least one transparent numeric impact path."
            )
        else:
            warning = None

        summary = f"Numerical grounding coverage: {('OK' if not missing else 'Missing ' + str(len(missing)) + ' component(s)')}."
        return NumericalGroundingCoverageReport(
            warning=warning,
            missing_components=missing[:5],
            summary=summary,
        )


class ThesisBalanceChecker:
    """
    Ensures synthesis covers structural, cyclical, temporary, macro, and fundamental
    drivers rather than anchoring on a single retrieved narrative theme.
    """

    _DRIVER_BUCKETS: Dict[str, List[str]] = {
        "structural": [
            "structural", "long-term", "secular", "competitive", "market structure",
            "moat", "industry", "regulatory framework", "supply chain",
        ],
        "cyclical": [
            "cyclical", "cycle", "seasonal", "inventory cycle", "demand cycle",
            "upturn", "downturn", "recession", "recovery",
        ],
        "temporary": [
            "temporary", "one-off", "short-term", "headline", "episodic",
            "this quarter", "this month", "breaking", "announcement",
        ],
        "macro": [
            "macro", "gdp", "inflation", "interest rate", "monetary", "fiscal",
            "geopolitical", "tariff", "currency", "central bank",
        ],
        "fundamentals": [
            "fundamental", "earnings", "revenue", "margin", "cash flow",
            "balance sheet", "valuation", "p/e", "ebitda", "unit economics",
        ],
    }

    _MARKET_CONTEXT_KEYWORDS = {
        "market", "valuation", "revenue", "stock", "earnings", "financial", "investment",
    }

    def check(
        self,
        draft: str,
        simulation_requirement: str,
        section_title: str,
        min_buckets: int = 3,
    ) -> ThesisBalanceReport:
        draft_clean = re.sub(r"<self_critique>.*?</self_critique>", "", draft, flags=re.DOTALL)
        context_lower = (simulation_requirement + " " + section_title).lower()
        text_lower = (context_lower + " " + draft_clean).lower()

        is_market_context = any(k in context_lower for k in self._MARKET_CONTEXT_KEYWORDS)
        required_buckets = min_buckets if is_market_context else 2

        present: List[str] = []
        missing: List[str] = []
        bucket_hits: Dict[str, int] = {}

        for bucket, keywords in self._DRIVER_BUCKETS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            bucket_hits[bucket] = hits
            if hits > 0:
                present.append(bucket)
            else:
                missing.append(bucket)

        dominant_bucket = None
        if bucket_hits:
            top_bucket, top_hits = max(bucket_hits.items(), key=lambda kv: kv[1])
            total_hits = sum(bucket_hits.values()) or 1
            if top_hits >= 3 and (top_hits / total_hits) > 0.55:
                dominant_bucket = top_bucket

        warning = None
        if len(present) < required_buckets:
            warning = (
                f"⚠️ THESIS IMBALANCE: Only {len(present)}/{len(self._DRIVER_BUCKETS)} driver classes "
                f"represented ({', '.join(present) or 'none'}). Missing: {', '.join(missing)}. "
                "Balance structural, cyclical, temporary, macro, and fundamental drivers — "
                "do not center the section on one retrieved headline theme."
            )
        elif dominant_bucket:
            warning = (
                f"⚠️ THESIS CONCENTRATION: Driver class '{dominant_bucket}' dominates the narrative. "
                "Explicitly integrate at least one contrasting driver (e.g., cyclical vs structural, "
                "temporary event vs fundamentals) before finalizing."
            )

        if warning:
            summary = f"Thesis balance: {len(present)} buckets present; dominant={dominant_bucket or 'none'}."
        else:
            summary = f"Thesis balance OK ({len(present)} driver classes represented)."

        return ThesisBalanceReport(
            buckets_present=present,
            buckets_missing=missing,
            dominant_bucket=dominant_bucket,
            warning=warning,
            summary=summary,
        )


class EpistemicReviewChecker:
    """
    Asks whether the draft over-synthesizes weak epistemic tiers into strategic certainty.
    """

    _CITATION_RE = re.compile(r"\[[Ss]\s*(\d+)\]")
    _CONFIDENT_KEYWORDS = {
        "will", "definitely", "clearly", "obviously", "certainly", "proves",
        "undoubtedly", "always", "must", "guaranteed",
    }
    _TENTATIVE_TIERS = {"RETAIL_COMMENTARY", "SOCIAL_OPINION", "SPECULATIVE"}
    _PRIMARY_TIERS = {"SIMULATION_PRIMARY", "INSTITUTIONAL", "PROFESSIONAL"}

    def check(self, draft: str, evidence_scores: Dict[int, Any]) -> EpistemicReviewReport:
        draft_clean = re.sub(r"<self_critique>.*?</self_critique>", "", draft, flags=re.DOTALL)
        citations = [int(m.group(1)) for m in self._CITATION_RE.finditer(draft_clean)]

        tier_rank_by_num: Dict[int, int] = {}
        tier_by_num: Dict[int, str] = {}
        for num, score in evidence_scores.items():
            tier = getattr(score, "epistemic_tier", "PROFESSIONAL")
            tier_by_num[num] = tier
            tier_rank_by_num[num] = EpistemicDiscriminator._TIER_META.get(tier, (2,))[0]

        low_tier_cites = 0
        primary_cites = 0
        for c in citations:
            tier = tier_by_num.get(c, "PROFESSIONAL")
            if tier in self._TENTATIVE_TIERS or tier == "ANALYST_ACTION":
                low_tier_cites += 1
            if tier in self._PRIMARY_TIERS:
                primary_cites += 1

        total_cites = len(citations) or 1
        low_tier_share = low_tier_cites / total_cites
        evidence_quality_collapse = low_tier_share > 0.45 and total_cites >= 4
        missing_primary_anchor = primary_cites == 0 and total_cites >= 3

        overconfident_low: List[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", draft_clean):
            sent = sent.strip()
            if len(sent) < 25:
                continue
            cited = {int(m.group(1)) for m in self._CITATION_RE.finditer(sent)}
            if not cited:
                continue
            sent_lower = sent.lower()
            if not any(w in sent_lower for w in self._CONFIDENT_KEYWORDS):
                continue
            worst_rank = max(tier_rank_by_num.get(n, 2) for n in cited)
            if worst_rank >= 4:
                overconfident_low.append(sent[:180] + ("…" if len(sent) > 180 else ""))

        critique_questions = [
            "Is each central claim supported by SIMULATION_PRIMARY, INSTITUTIONAL, or PROFESSIONAL evidence?",
            "Are LinkedIn comments, retail media, or analyst target changes relegated to tentative context only?",
            "Is any conclusion overstated relative to the weakest cited epistemic tier?",
            "Are alternative explanations and counter-scenarios explicitly stated?",
            "Is the section overfitting to one salient retrieved narrative?",
        ]

        warnings: List[str] = []
        if evidence_quality_collapse:
            warnings.append(
                f"Evidence quality collapse: {low_tier_share:.0%} of citations are "
                "retail/social/speculative/analyst-only sources"
            )
        if missing_primary_anchor:
            warnings.append(
                "No primary-eligible epistemic tier (simulation/institutional/professional) cited — "
                "thesis may rest on weak commentary"
            )
        if overconfident_low:
            warnings.append(
                f"{len(overconfident_low)} confident claim(s) cite only low-epistemic-tier sources"
            )

        warning = None
        if warnings:
            warning = "⚠️ EPISTEMIC REVIEW: " + "; ".join(warnings)

        summary = (
            f"Epistemic review: low-tier cite share {low_tier_share:.0%}, "
            f"primary anchors={'yes' if primary_cites else 'no'}, "
            f"overconfident low-tier={len(overconfident_low)}"
        )

        return EpistemicReviewReport(
            evidence_quality_collapse=evidence_quality_collapse,
            overconfident_low_tier_sentences=overconfident_low[:5],
            missing_primary_anchor=missing_primary_anchor,
            low_tier_citation_share=low_tier_share,
            critique_questions=critique_questions,
            warning=warning,
            summary=summary,
        )
