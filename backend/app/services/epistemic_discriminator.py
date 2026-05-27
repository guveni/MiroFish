"""
Epistemic discrimination: classify retrieved evidence by trust tier and signal vs noise.

Separates simulation ground truth, institutional filings, professional journalism,
analyst actions, retail commentary, and social opinion so synthesis cannot treat
them at equal narrative weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class EpistemicClassification:
    """Epistemic tier and synthesis constraints for one evidence source."""

    tier: str
    tier_rank: int  # 0 = highest trust, higher = lower trust
    credibility_cap: float
    confidence_ceiling: str  # HIGH | MEDIUM | LOW | TENTATIVE_ONLY
    max_thesis_share: float  # recommended max share of section claims citing this source
    signals: Tuple[str, ...] = field(default_factory=tuple)
    display_label: str = ""

    @property
    def is_primary_eligible(self) -> bool:
        return self.tier_rank <= 2

    @property
    def is_tentative_only(self) -> bool:
        return self.tier_rank >= 4 or self.confidence_ceiling == "TENTATIVE_ONLY"


class EpistemicDiscriminator:
    """
    Content- and domain-aware epistemic tiering.

    Tiers (high → low):
      SIMULATION_PRIMARY — interview_agents / direct simulation ground truth
      INSTITUTIONAL — filings, regulators, tier-1 wire services
      PROFESSIONAL — graph retrieval, established business press
      ANALYST_ACTION — rating/target changes without full model support
      RETAIL_COMMENTARY — retail investor media, generic market blogs
      SOCIAL_OPINION — LinkedIn, comments, forums, viral reactions
      SPECULATIVE — rumors, unconfirmed reports
    """

    TIER_ORDER = [
        "SIMULATION_PRIMARY",
        "INSTITUTIONAL",
        "PROFESSIONAL",
        "ANALYST_ACTION",
        "RETAIL_COMMENTARY",
        "SOCIAL_OPINION",
        "SPECULATIVE",
    ]

    _TIER_META = {
        "SIMULATION_PRIMARY": (0, 1.0, "HIGH", 0.45),
        "INSTITUTIONAL": (1, 0.92, "HIGH", 0.40),
        "PROFESSIONAL": (2, 0.78, "MEDIUM", 0.35),
        "ANALYST_ACTION": (3, 0.52, "LOW", 0.20),
        "RETAIL_COMMENTARY": (4, 0.32, "TENTATIVE_ONLY", 0.12),
        "SOCIAL_OPINION": (5, 0.18, "TENTATIVE_ONLY", 0.08),
        "SPECULATIVE": (6, 0.12, "TENTATIVE_ONLY", 0.08),
    }

    _TOOL_BASE_TIER = {
        "interview_agents": "SIMULATION_PRIMARY",
        "insight_forge": "SIMULATION_PRIMARY",
        "panorama_search": "PROFESSIONAL",
        "quick_search": "PROFESSIONAL",
        "web_search": "RETAIL_COMMENTARY",
    }

    _INSTITUTIONAL_DOMAINS = {
        "sec.gov", "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "economist.com", "ftc.gov", "fda.gov", "europa.eu", "imf.org",
        "worldbank.org", "oecd.org", "federalreserve.gov", "bis.gov",
        "justice.gov", "treasury.gov", "whitehouse.gov",
    }

    _RETAIL_DOMAINS = {
        "fool.com", "motleyfool.com", "investorplace.com", "benzinga.com",
        "seekingalpha.com", "247wallst.com", "zacks.com", "tipranks.com",
        "marketwatch.com", "investopedia.com", "yahoo.com",
    }

    _SOCIAL_DOMAINS = {
        "linkedin.com", "reddit.com", "twitter.com", "x.com", "facebook.com",
        "tiktok.com", "stocktwits.com", "discord.com", "medium.com",
    }

    _INSTITUTIONAL_PHRASES = [
        "10-k", "10-q", "8-k", "sec filing", "form 10", "earnings release",
        "investor relations", "annual report", "proxy statement", "regulatory filing",
        "consolidated financial", "audited financial",
    ]

    _ANALYST_PHRASES = [
        "upgraded to", "downgraded to", "price target", "rating change",
        "maintains buy", "maintains sell", "outperform", "underperform",
        "initiated coverage", "raises target", "cuts target", "analyst at",
        "consensus estimate", "street expects",
    ]

    _SOCIAL_PHRASES = [
        "linkedin", "comment on", "comments section", "posted on twitter",
        "tweeted", "reddit user", "forum post", "social media reaction",
        "viral post", "influencer", "user commented",
    ]

    _RETAIL_PHRASES = [
        "retail investor", "stock pick", "hot stock", "must-buy",
        "wallstreetbets", "yolo", "meme stock",
    ]

    _SPECULATIVE_PHRASES = [
        "rumor", "rumour", "reportedly", "sources say", "unconfirmed",
        "speculation that", "might be considering", "in talks to",
        "people familiar", "according to sources", "leaked",
    ]

    def classify(self, raw_result: str, tool_name: str) -> EpistemicClassification:
        """
        Assign epistemic tier: tool sets baseline; content upgrades or downgrades.

        Downgrade signals (social, retail, rumor) pull trust down.
        Upgrade signals (filings, tier-1 domains, agent quotes) pull trust up.
        """
        text_lower = raw_result.lower()
        domains = self._extract_domains(raw_result)

        base_tier = self._TOOL_BASE_TIER.get(tool_name, "PROFESSIONAL")
        tier_rank = self._TIER_META[base_tier][0]
        signals: List[str] = [f"tool:{tool_name}"]

        def _apply(tier: str, reason: str, *, downgrade: bool) -> None:
            nonlocal tier_rank
            rank = self._TIER_META[tier][0]
            if downgrade:
                if rank > tier_rank:
                    tier_rank = rank
                    signals.append(reason)
            else:
                if rank < tier_rank:
                    tier_rank = rank
                    signals.append(reason)

        for domain in domains:
            if any(inst in domain for inst in self._INSTITUTIONAL_DOMAINS):
                _apply("INSTITUTIONAL", f"domain:{domain}", downgrade=False)
            if any(soc in domain for soc in self._SOCIAL_DOMAINS):
                _apply("SOCIAL_OPINION", f"domain:{domain}", downgrade=True)
            if any(ret in domain for ret in self._RETAIL_DOMAINS):
                _apply("RETAIL_COMMENTARY", f"domain:{domain}", downgrade=True)

        for phrase in self._INSTITUTIONAL_PHRASES:
            if phrase in text_lower:
                _apply("INSTITUTIONAL", f"phrase:{phrase}", downgrade=False)
                break

        for phrase in self._ANALYST_PHRASES:
            if phrase in text_lower:
                _apply("ANALYST_ACTION", f"phrase:{phrase}", downgrade=True)

        for phrase in self._SOCIAL_PHRASES:
            if phrase in text_lower:
                _apply("SOCIAL_OPINION", f"phrase:{phrase}", downgrade=True)

        for phrase in self._RETAIL_PHRASES:
            if phrase in text_lower:
                _apply("RETAIL_COMMENTARY", f"phrase:{phrase}", downgrade=True)

        for phrase in self._SPECULATIVE_PHRASES:
            if phrase in text_lower:
                _apply("SPECULATIVE", f"phrase:{phrase}", downgrade=True)

        if tool_name in ("insight_forge", "panorama_search", "quick_search", "interview_agents"):
            if re.search(r'^>\s*"|interview|agent said|simulated', text_lower, re.MULTILINE):
                _apply("SIMULATION_PRIMARY", "simulation:agent_quote", downgrade=False)

        final_tier = self.TIER_ORDER[tier_rank]
        return self._build_classification(final_tier, tuple(signals[:8]))

    def _build_classification(self, tier: str, signals: Tuple[str, ...]) -> EpistemicClassification:
        rank, cap, ceiling, max_share = self._TIER_META[tier]
        labels = {
            "SIMULATION_PRIMARY": "Simulation / agent ground truth",
            "INSTITUTIONAL": "Institutional-grade evidence",
            "PROFESSIONAL": "Professional / graph retrieval",
            "ANALYST_ACTION": "Analyst rating or target change (secondary)",
            "RETAIL_COMMENTARY": "Retail financial commentary",
            "SOCIAL_OPINION": "Social / comment opinion",
            "SPECULATIVE": "Speculative / unconfirmed",
        }
        return EpistemicClassification(
            tier=tier,
            tier_rank=rank,
            credibility_cap=cap,
            confidence_ceiling=ceiling,
            max_thesis_share=max_share,
            signals=signals,
            display_label=labels.get(tier, tier),
        )

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

    def build_hierarchy_block(self, source_scores: List[Tuple[int, str, EpistemicClassification]]) -> str:
        """Human-readable epistemic hierarchy for pre-synthesis injection."""
        if not source_scores:
            return ""

        sorted_sources = sorted(source_scores, key=lambda x: x[2].tier_rank)
        lines = [
            "=================================================================",
            "[EPISTEMIC HIERARCHY — DO NOT BLEND TIERS AT EQUAL WEIGHT]",
            "=================================================================",
            "Primary thesis anchors (use for central claims):",
        ]

        primary = [s for s in sorted_sources if s[2].is_primary_eligible]
        tentative = [s for s in sorted_sources if s[2].is_tentative_only]

        if primary:
            for num, tool, ep in primary:
                lines.append(
                    f"  • [S{num}] {ep.display_label} ({tool}) — ceiling: {ep.confidence_ceiling}"
                )
        else:
            lines.append("  • (none detected — keep claims hedged and simulation-grounded)")

        lines.append("")
        lines.append("Tentative-only (sentiment/context; NOT strategic conclusions):")
        if tentative:
            for num, tool, ep in tentative:
                lines.append(
                    f"  • [S{num}] {ep.display_label} ({tool}) — max ~{ep.max_thesis_share:.0%} of claims"
                )
        else:
            lines.append("  • (none)")

        lines.extend([
            "",
            "Rules:",
            "- Never treat LinkedIn comments, retail blogs, or analyst target changes as institutional proof.",
            "- Label social/retail/speculative citations: '(tentative sentiment, low epistemic weight)'.",
            "- Structural drivers must come from SIMULATION_PRIMARY / INSTITUTIONAL / PROFESSIONAL tiers.",
            "=================================================================",
        ])
        return "\n".join(lines)
