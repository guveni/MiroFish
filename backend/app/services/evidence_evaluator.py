import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse

from .epistemic_discriminator import EpistemicDiscriminator, EpistemicClassification

@dataclass
class EvidenceScore:
    source_num: int
    tool_name: str
    credibility: float
    relevance: float
    recency: float
    specificity: float
    strategic_materiality: float
    synthesis_weight: float
    key_claims_count: int
    quantified_claims_count: int
    speculative_claims_count: int
    durability: float = 0.5
    epistemic_tier: str = "PROFESSIONAL"
    confidence_ceiling: str = "MEDIUM"
    epistemic_label: str = ""

    @property
    def weight_band(self) -> str:
        if self.synthesis_weight >= 0.8:
            return "HIGH"
        elif self.synthesis_weight >= 0.55:
            return "MEDIUM-HIGH"
        elif self.synthesis_weight >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def is_tentative_only(self) -> bool:
        return self.confidence_ceiling == "TENTATIVE_ONLY" or self.epistemic_tier in (
            "RETAIL_COMMENTARY", "SOCIAL_OPINION", "SPECULATIVE",
        )


@dataclass(frozen=True)
class SectionConfidenceProfile:
    """Aggregate confidence posture for a section — drives synthesis behavior."""

    section_ceiling: str  # HIGH | MEDIUM | LOW | TENTATIVE_ONLY
    aggregate_synthesis_strength: float
    primary_eligible_count: int
    tentative_weight_share: float
    max_confident_sentence_ratio: float
    assertion_strength: str  # definitive | balanced | hedged | exploratory
    recommendation_mode: str  # assertive | conditional | observational


class EvidenceEvaluator:
    """
    Evaluates retrieved evidence with epistemic discrimination before synthesis.
    """

    def __init__(self) -> None:
        self._epistemic = EpistemicDiscriminator()

    _TOOL_CREDIBILITY = {
        "interview_agents": 1.0,
        "insight_forge": 1.0,
        "panorama_search": 0.7,
        "quick_search": 0.7,
        "web_search": 0.4,
    }

    _LOW_QUALITY_WEB_DOMAINS = {
        "reddit.com", "twitter.com", "x.com", "facebook.com", "tiktok.com",
        "linkedin.com", "medium.com", "substack.com", "seekingalpha.com", "fool.com",
        "benzinga.com", "zerohedge.com", "stocktwits.com",
    }

    _HIGH_QUALITY_WEB_DOMAINS = {
        "sec.gov", "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "economist.com", "ftc.gov", "fda.gov", "europa.eu", "imf.org",
        "worldbank.org", "oecd.org", "federalreserve.gov", "bis.org",
    }

    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "why", "who", "which", "this", "that", "these",
        "those", "with", "from", "into", "over", "under", "about", "for", "to",
    }

    _MATERIALITY_KEYWORDS = [
        "revenue", "profit", "loss", "cost", "dilution", "accretion", "market size",
        "valuation", "market share", "gdp", "capital", "funding", "pricing", "tariff",
        "margin", "ebitda", "sales", "earnings", "user base", "deadline", "penalty",
        "fines", "compliance cost", "usd", "billion", "million", "acquisition",
    ]

    _SPECULATIVE_KEYWORDS = [
        "maybe", "perhaps", "possibly", "potential", "could", "might", "may",
        "suggests", "scenario", "hypothetically", "outlook", "prospects",
    ]

    _DURABLE_SIGNALS = [
        "structural", "long-term", "secular", "competitive position", "market structure",
        "regulatory framework", "compliance regime", "tam", "addressable market",
        "moat", "unit economics", "balance sheet", "cash flow", "fundamentals",
        "supply chain structure", "industry concentration",
    ]

    _EPHEMERAL_SIGNALS = [
        "breaking", "headline", "rumor", "reportedly", "sources say", "talks",
        "merger talks", "deal talks", "spike", "surge", "plunge", "today",
        "this week", "flash", "viral", "tweet", "leaked",
    ]

    _CEILING_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "TENTATIVE_ONLY": 0}
    _RANK_TO_CEILING = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "TENTATIVE_ONLY"}

    def score_evidence(
        self,
        source_num: int,
        raw_result: str,
        tool_name: str,
        section_title: str,
        simulation_requirement: str,
        freshness_score: float = 0.7,
    ) -> EvidenceScore:
        """Calculate hierarchical quality metrics with epistemic tier caps."""
        epistemic = self._epistemic.classify(raw_result, tool_name)

        credibility = min(
            self._TOOL_CREDIBILITY.get(tool_name, 0.4),
            epistemic.credibility_cap,
        )
        if tool_name == "web_search":
            credibility = min(credibility, self._adjust_web_credibility(raw_result, credibility))

        relevance = self._calculate_relevance(raw_result, section_title, simulation_requirement)
        recency = freshness_score
        specificity, key_claims, quantified, speculative = self._calculate_specificity_and_claims(raw_result)
        materiality = self._calculate_materiality(raw_result)
        durability = self._calculate_durability(raw_result, tool_name, epistemic)

        total_claims = max(1, key_claims + quantified + speculative)
        speculative_ratio = speculative / total_claims
        speculative_penalty = min(0.12, speculative_ratio * 0.12)

        synthesis_weight = (
            credibility * 0.30
            + durability * 0.18
            + materiality * 0.17
            + specificity * 0.15
            + relevance * 0.10
            + recency * 0.08
        ) - speculative_penalty

        # Hard cap: epistemic tier cannot be overridden by retrieval salience
        synthesis_weight = min(synthesis_weight, epistemic.credibility_cap)
        synthesis_weight = max(0.05, min(1.0, synthesis_weight))

        return EvidenceScore(
            source_num=source_num,
            tool_name=tool_name,
            credibility=round(credibility, 2),
            relevance=round(relevance, 2),
            recency=round(recency, 2),
            specificity=round(specificity, 2),
            strategic_materiality=round(materiality, 2),
            synthesis_weight=round(synthesis_weight, 2),
            key_claims_count=key_claims,
            quantified_claims_count=quantified,
            speculative_claims_count=speculative,
            durability=round(durability, 2),
            epistemic_tier=epistemic.tier,
            confidence_ceiling=epistemic.confidence_ceiling,
            epistemic_label=epistemic.display_label,
        )

    def _adjust_web_credibility(self, text: str, base: float) -> float:
        domains = self._extract_domains(text)
        if not domains:
            return base
        if any(d in self._HIGH_QUALITY_WEB_DOMAINS for d in domains):
            return min(1.0, base + 0.15)
        if any(any(low in d for low in self._LOW_QUALITY_WEB_DOMAINS) for d in domains):
            return max(0.10, base - 0.12)
        return base

    @staticmethod
    def _extract_domains(text: str) -> List[str]:
        domains: List[str] = []
        for url in re.findall(r"https?://[^\s\])>\"']+", text):
            try:
                host = urlparse(url).netloc.lower().replace("www.", "")
                if host:
                    domains.append(host)
            except Exception:
                continue
        return domains

    def _calculate_relevance(self, text: str, section_title: str, simulation_requirement: str) -> float:
        text_lower = text.lower()
        combined_topic = f"{section_title} {simulation_requirement}".lower()
        keywords = set(re.findall(r"\b[a-z]{4,}\b", combined_topic))
        keywords = {w for w in keywords if w not in self._STOP_WORDS}
        if not keywords:
            return 0.5
        matches = sum(1 for kw in keywords if kw in text_lower)
        return min(1.0, matches / len(keywords))

    def _calculate_durability(
        self, text: str, tool_name: str, epistemic: EpistemicClassification
    ) -> float:
        text_lower = text.lower()
        durable_hits = sum(1 for kw in self._DURABLE_SIGNALS if kw in text_lower)
        ephemeral_hits = sum(1 for kw in self._EPHEMERAL_SIGNALS if kw in text_lower)

        score = 0.45
        if durable_hits >= 2:
            score += 0.35
        elif durable_hits == 1:
            score += 0.2
        if ephemeral_hits >= 2:
            score -= 0.35
        elif ephemeral_hits == 1:
            score -= 0.2
        if tool_name in ("interview_agents", "insight_forge", "panorama_search"):
            score += 0.1
        if bool(re.search(r"\b\d+(?:\.\d+)?%|\$\s*\d", text_lower)):
            score += 0.08

        # Social/retail/speculative content is structurally low-durability
        if epistemic.tier_rank >= 4:
            score = min(score, 0.35)
        elif epistemic.tier == "ANALYST_ACTION":
            score = min(score, 0.55)

        return max(0.1, min(1.0, score))

    def _calculate_specificity_and_claims(self, text: str) -> Tuple[float, int, int, int]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
        if not sentences:
            return 0.0, 0, 0, 0

        key_claims = 0
        quantified = 0
        speculative = 0
        pct_re = re.compile(r"\b\d+(?:\.\d+)?%")
        num_re = re.compile(r"\b\d[\d,]*\.?\d*\b")
        entity_re = re.compile(r"\b[A-Z][a-zA-Z0-9_]+\b")
        date_re = re.compile(r"\b20[1-3]\d\b")
        quote_re = re.compile(r'"[^"]+"|\'[^\']+\'|^>')

        for sent in sentences:
            sent_lower = sent.lower()
            is_factual = False
            has_num = bool(num_re.search(sent)) or "$" in sent
            has_pct = bool(pct_re.search(sent))
            has_entity = len(entity_re.findall(sent)) >= 2
            has_date = bool(date_re.search(sent))
            has_quote = bool(quote_re.search(sent)) or sent.startswith(">")

            if has_num or has_pct:
                quantified += 1
                is_factual = True
            elif has_entity or has_date or has_quote:
                key_claims += 1
                is_factual = True
            if any(w in sent_lower for w in self._SPECULATIVE_KEYWORDS):
                speculative += 1
                if is_factual:
                    is_factual = False

        total_claims = key_claims + quantified
        specificity = min(1.0, (total_claims * 1.5 + speculative * 0.5) / max(1, len(sentences)))
        return specificity, total_claims, quantified, speculative

    def _calculate_materiality(self, text: str) -> float:
        text_lower = text.lower()
        matched_kws = sum(1 for kw in self._MATERIALITY_KEYWORDS if kw in text_lower)
        has_quantifier = bool(re.search(r"\b\d+(?:\.\d+)?%|\$\s*\d", text_lower))
        if matched_kws == 0:
            return 0.0
        if matched_kws == 1:
            return 0.3 if has_quantifier else 0.15
        if matched_kws == 2:
            return 0.6 if has_quantifier else 0.4
        return 1.0 if has_quantifier else 0.8

    def build_evidence_card(self, score: EvidenceScore, raw_result: str) -> str:
        advice = []
        advice.append(
            f"Epistemic tier: {score.epistemic_tier} — {score.epistemic_label} "
            f"(confidence ceiling: {score.confidence_ceiling})"
        )
        if score.is_tentative_only:
            advice.append(
                "⚠ TENTATIVE ONLY — social/retail/speculative source. "
                "Do NOT use for central thesis or strategic conclusions."
            )
        if score.epistemic_tier == "ANALYST_ACTION":
            advice.append(
                "⚠ Analyst rating/target change — secondary signal only; "
                "require mechanism + quantified impact before weighting."
            )
        if score.strategic_materiality < 0.3:
            advice.append("⚠ Low materiality — no quantified business/financial impact.")
        if score.durability < 0.4:
            advice.append("⚠ Low durability — episodic/headline noise vs structural driver.")
        if score.recency < 0.5:
            advice.append("⚠ Stale — add 'as of [date]' or verify currency.")
        if score.speculative_claims_count > score.quantified_claims_count:
            advice.append("⚠ Speculation-heavy — hedge unless cross-verified at INSTITUTIONAL tier.")

        advice_str = "\n".join(f"  {a}" for a in advice)

        lines = [
            f"[Evidence Card — {score.tool_name} — S{score.source_num}]",
            (
                f"Tier: {score.epistemic_tier} | Cred: {score.credibility:.1f} | "
                f"Dur: {score.durability:.1f} | Rec: {score.recency:.1f}"
            ),
            (
                f"Spec: {score.specificity:.1f} | Mat: {score.strategic_materiality:.1f} | "
                f"Weight: {score.weight_band} ({score.synthesis_weight:.2f})"
            ),
            "Epistemic guidance:",
            advice_str,
            "---",
        ]
        return "\n".join(lines) + f"\n\n{raw_result}"

    def detect_concentration(self, all_scores: List[EvidenceScore]) -> Optional[str]:
        if len(all_scores) < 3:
            return None
        total_weight = sum(s.synthesis_weight for s in all_scores)
        if total_weight <= 0:
            return None
        for score in all_scores:
            share = score.synthesis_weight / total_weight
            if share > 0.40:
                tier_note = f", tier={score.epistemic_tier}" if score.is_tentative_only else ""
                return (
                    f"[Evidence Concentration Warning] [S{score.source_num}] ({score.tool_name}) "
                    f"is {share:.1%} of weight{tier_note}. "
                    "Risk of anchoring on salient low-epistemic retrieval — broaden tools or downrank."
                )
        return None

    def build_digest(self, all_scores: List[EvidenceScore]) -> str:
        if not all_scores:
            return "[Evidence Digest]\nNo retrieved evidence available."

        sorted_scores = sorted(all_scores, key=lambda s: s.synthesis_weight, reverse=True)
        tier_groups: Dict[str, List[EvidenceScore]] = {}
        for s in all_scores:
            tier_groups.setdefault(s.epistemic_tier, []).append(s)

        lines = [
            "=================================================================",
            "[EVIDENCE DIGEST — EPISTEMIC HIERARCHY FIRST, THEN WEIGHT]",
            "=================================================================",
            "Order of trust (never blend at equal weight):",
            "  SIMULATION_PRIMARY > INSTITUTIONAL > PROFESSIONAL > ANALYST_ACTION",
            "  > RETAIL_COMMENTARY > SOCIAL_OPINION > SPECULATIVE",
            "",
        ]

        tier_order = EpistemicDiscriminator.TIER_ORDER
        for tier in tier_order:
            group = tier_groups.get(tier, [])
            if not group:
                continue
            lines.append(f"--- {tier} ---")
            for s in sorted(group, key=lambda x: x.synthesis_weight, reverse=True):
                role = "PRIMARY-ELIGIBLE" if not s.is_tentative_only else "TENTATIVE-ONLY"
                lines.append(
                    f"  [S{s.source_num}] {s.tool_name} weight={s.synthesis_weight:.2f} "
                    f"({role}, ceiling={s.confidence_ceiling})"
                )
            lines.append("")

        lines.append("Weighted ranking (for tie-break within tier):")
        for rank, score in enumerate(sorted_scores, 1):
            lines.append(
                f"  {rank}. [S{score.source_num}] {score.epistemic_tier} — "
                f"{score.weight_band} ({score.synthesis_weight:.2f})"
            )

        tentative = [s for s in sorted_scores if s.is_tentative_only]
        if tentative:
            ids = ", ".join(f"S{s.source_num}" for s in tentative)
            lines.append(f"\n[Downrank] {ids} — context/sentiment only, not thesis anchors.")

        lines.append("=================================================================")
        return "\n".join(lines)

    def build_synthesis_mandate(
        self,
        all_scores: List[EvidenceScore],
        section_title: str = "",
    ) -> str:
        if not all_scores:
            return ""

        sorted_scores = sorted(all_scores, key=lambda s: s.synthesis_weight, reverse=True)
        primary_eligible = [s for s in sorted_scores if not s.is_tentative_only]
        tentative = [s for s in sorted_scores if s.is_tentative_only]

        lines = [
            "=================================================================",
            "[PRE-SYNTHESIS EPISTEMIC MANDATE — MANDATORY]",
            "=================================================================",
            f"Section: {section_title or '(current section)'}",
            "",
            "Epistemic discrimination rules:",
            "- Central claims MUST cite SIMULATION_PRIMARY / INSTITUTIONAL / PROFESSIONAL sources.",
            "- ANALYST_ACTION: secondary only (rating/target); add mechanism + numbers.",
            "- SOCIAL_OPINION / RETAIL_COMMENTARY / SPECULATIVE: label tentative; max ~10% of claims.",
            "- Never convert plausible commentary into strategic certainty.",
            "- Separate structural drivers from temporary reactions and social sentiment.",
            "",
        ]

        if primary_eligible:
            anchors = ", ".join(
                f"S{s.source_num}({s.epistemic_tier})"
                for s in primary_eligible[:3]
            )
            lines.append(f"Permitted thesis anchors: {anchors}")
        else:
            lines.append("Permitted thesis anchors: none — keep section hedged and simulation-grounded.")

        if tentative:
            lines.append(
                "Tentative-only: "
                + ", ".join(f"S{s.source_num}({s.epistemic_tier})" for s in tentative)
            )

        top = sorted_scores[0]
        if top.is_tentative_only:
            lines.append(
                f"⚠ Highest-weight source [S{top.source_num}] is {top.epistemic_tier} — "
                "do NOT center the section on this retrieval."
            )

        concentration = self.detect_concentration(all_scores)
        if concentration:
            lines.append(concentration)

        profile = self.compute_section_profile(all_scores)
        lines.append(self.build_confidence_governed_block(profile))
        lines.append(self.build_narrative_weight_allocation(all_scores))

        lines.append("=================================================================")
        return "\n".join(lines)

    def compute_section_profile(self, all_scores: List[EvidenceScore]) -> SectionConfidenceProfile:
        if not all_scores:
            return SectionConfidenceProfile(
                section_ceiling="LOW",
                aggregate_synthesis_strength=0.0,
                primary_eligible_count=0,
                tentative_weight_share=1.0,
                max_confident_sentence_ratio=0.15,
                assertion_strength="exploratory",
                recommendation_mode="observational",
            )

        total_weight = sum(s.synthesis_weight for s in all_scores) or 1.0
        primary = [s for s in all_scores if not s.is_tentative_only]
        tentative_share = (
            sum(s.synthesis_weight for s in all_scores if s.is_tentative_only) / total_weight
        )

        if primary:
            weighted_rank = sum(
                self._CEILING_RANK.get(s.confidence_ceiling, 1) * s.synthesis_weight
                for s in primary
            ) / sum(s.synthesis_weight for s in primary)
            section_rank = int(round(weighted_rank))
        else:
            section_rank = 0

        section_rank = max(0, min(3, section_rank))
        if tentative_share > 0.55 and section_rank > 1:
            section_rank = min(section_rank, 1)
        if tentative_share > 0.75:
            section_rank = 0

        ceiling = self._RANK_TO_CEILING[section_rank]
        agg_strength = sum(s.synthesis_weight for s in all_scores) / len(all_scores)

        if ceiling == "HIGH":
            assertion, rec_mode, max_conf_ratio = "balanced", "assertive", 0.40
        elif ceiling == "MEDIUM":
            assertion, rec_mode, max_conf_ratio = "balanced", "conditional", 0.28
        elif ceiling == "LOW":
            assertion, rec_mode, max_conf_ratio = "hedged", "conditional", 0.18
        else:
            assertion, rec_mode, max_conf_ratio = "exploratory", "observational", 0.08

        return SectionConfidenceProfile(
            section_ceiling=ceiling,
            aggregate_synthesis_strength=round(agg_strength, 2),
            primary_eligible_count=len(primary),
            tentative_weight_share=round(tentative_share, 2),
            max_confident_sentence_ratio=max_conf_ratio,
            assertion_strength=assertion,
            recommendation_mode=rec_mode,
        )

    def build_confidence_governed_block(self, profile: SectionConfidenceProfile) -> str:
        banned = []
        if profile.section_ceiling in ("LOW", "TENTATIVE_ONLY"):
            banned = ["will", "definitely", "clearly", "certainly", "proves", "undoubtedly", "guaranteed"]
        elif profile.section_ceiling == "MEDIUM":
            banned = ["definitely", "proves", "undoubtedly", "guaranteed", "always"]

        lines = [
            "",
            "[CONFIDENCE-GOVERNED SYNTHESIS — BEHAVIORAL RULES]",
            f"Section confidence ceiling: {profile.section_ceiling}",
            f"Assertion strength: {profile.assertion_strength}",
            f"Recommendation mode: {profile.recommendation_mode}",
            f"Max share of definitive sentences (will/definitely/proves): ≤{profile.max_confident_sentence_ratio:.0%}",
        ]
        if profile.primary_eligible_count == 0:
            lines.append(
                "No primary-eligible evidence — treat section as exploratory; "
                "no strategic recommendations; label all forward views as highly speculative."
            )
        if profile.tentative_weight_share > 0.4:
            lines.append(
                f"Tentative-source weight share {profile.tentative_weight_share:.0%} — "
                "downrank social/retail/speculative signals in conclusions."
            )
        if banned:
            lines.append(
                "Banned phrasing for non-simulation-primary claims: "
                + ", ".join(f'"{w}"' for w in banned)
            )
        lines.append(
            "Map language to ceiling: HIGH → 'likely/moderate confidence'; "
            "MEDIUM → 'may/moderately likely'; LOW → 'might/uncertain'; "
            "TENTATIVE_ONLY → 'speculative/tentative only'."
        )
        return "\n".join(lines)

    def build_narrative_weight_allocation(self, all_scores: List[EvidenceScore]) -> str:
        if not all_scores:
            return "[Narrative weight allocation]\nNo sources — do not invent thesis anchors."

        total = sum(s.synthesis_weight for s in all_scores) or 1.0
        lines = [
            "",
            "[NARRATIVE WEIGHT ALLOCATION — synthesis_weight controls emphasis, not salience]",
            "Allocate analytical emphasis proportionally (do not let one headline dominate):",
        ]
        for s in sorted(all_scores, key=lambda x: x.synthesis_weight, reverse=True):
            share = s.synthesis_weight / total
            cap = s.confidence_ceiling
            role = "thesis-eligible" if not s.is_tentative_only else "context-only"
            lines.append(
                f"  [S{s.source_num}] {s.epistemic_tier} — target ~{share:.0%} of section emphasis "
                f"({role}, ceiling={cap}, weight={s.synthesis_weight:.2f})"
            )
        top = max(all_scores, key=lambda x: x.synthesis_weight)
        if top.durability < 0.45 and top.synthesis_weight / total > 0.35:
            lines.append(
                f"⚠ [S{top.source_num}] is episodic/low-durability but high weight — "
                "frame as temporary driver, not structural thesis."
            )
        return "\n".join(lines)

    def build_epistemic_hierarchy(self, all_scores: List[EvidenceScore]) -> str:
        """Epistemic tier block for injection into ReACT observations."""
        if not all_scores:
            return ""
        entries: List[Tuple[int, str, EpistemicClassification]] = []
        for s in all_scores:
            rank, cap, ceiling, max_share = EpistemicDiscriminator._TIER_META[s.epistemic_tier]
            ep = EpistemicClassification(
                tier=s.epistemic_tier,
                tier_rank=rank,
                credibility_cap=cap,
                confidence_ceiling=ceiling,
                max_thesis_share=max_share,
                signals=(s.epistemic_label,),
                display_label=s.epistemic_label or s.epistemic_tier,
            )
            entries.append((s.source_num, s.tool_name, ep))
        return self._epistemic.build_hierarchy_block(entries)
