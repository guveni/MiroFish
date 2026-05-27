import re
from typing import List, Tuple
from . import BaseAnalyticalModule

class MarketsAnalyticalModule(BaseAnalyticalModule):
    """Domain-specific analytical rules for markets and finance."""
    
    def get_vague_phrases(self) -> List[str]:
        return ["market will decide", "investor sentiment is key", "financial prospects are good", "unclear financial impact"]
        
    def check_numerical_sanity(self, text: str, sources_metadata: List[Tuple[int, str, str]]) -> List[str]:
        anomalies = []
        import re
        pe_matches = re.finditer(r'\bp/e\b\s*(?:ratio)?\s*(?:of)?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        for pm in pe_matches:
            val = float(pm.group(1))
            if val > 1000.0:
                anomalies.append(f"Suspicious P/E ratio value of {val} detected.")
                
        move_matches = re.finditer(r'\b(\d+(?:\.\d+)?)\s*%\s*(?:surge|plunge|drop|increase|decrease|jump|fall|gain|loss|rise)\b', text, re.IGNORECASE)
        for mm in move_matches:
            val = float(mm.group(1))
            if val > 500.0:
                anomalies.append(f"Highly suspect market movement figure of {val}% detected (exceeds typical 500% daily market threshold).")
                
        return anomalies
        
    def check_quantitative_grounding(self, text: str, sources_metadata: List[Tuple[int, str, str]]) -> List[str]:
        anomalies = []
        text_lower = text.lower()

        financial_terms = ["revenue", "valuation", "earnings", "market cap", "profit", "margin", "p/e"]

        has_financial_term = any(term in text_lower for term in financial_terms)
        has_numerical_grounding = bool(
            re.search(
                r'\b\d+(?:\.\d+)?\s*(?:%|\bmillion\b|\bbillion\b|\bthousand\b)?|\$\s*\d+',
                text_lower,
            )
        )

        if has_financial_term and not has_numerical_grounding:
            anomalies.append(
                "Market/Financial analysis mentions terms like revenue, valuation, or earnings "
                "but lacks any concrete quantitative figures, market sizing, or monetary values."
            )

        merger_terms = ("merger", "acquisition", "m&a", "takeover", "buyout", "deal")
        if any(t in text_lower for t in merger_terms):
            depth_signals = {
                "mechanism": ("synergy", "integration", "premium", "consideration", "exchange ratio"),
                "dilution_accretion": ("dilution", "accretion", "eps accretion", "eps dilution"),
                "valuation": ("valuation", "p/e", "ev/ebitda", "multiple", "dcf", "enterprise value"),
                "economics": ("margin", "revenue", "cost of capital", "wacc", "fcf"),
            }
            missing_lenses = [
                label
                for label, kws in depth_signals.items()
                if not any(kw in text_lower for kw in kws)
            ]
            if len(missing_lenses) >= 2:
                anomalies.append(
                    "M&A discussion lacks domain depth — add mechanism, dilution/accretion, "
                    f"and valuation lenses (missing: {', '.join(missing_lenses[:3])})."
                )

        if "probability" not in text_lower and "scenario" in text_lower:
            if not re.search(r"\b\d+(?:\.\d+)?\s*%\s*(?:probability|likely|chance)", text_lower):
                anomalies.append(
                    "Scenario block present but lacks probability-weighted ranges "
                    "(e.g., Base 55–65%, Bear 20–30%)."
                )

        return anomalies

    def get_causal_chain_hints(self) -> str:
        return (
            "Market Specific: Link price movements to supply/demand, P/E or EV/EBITDA impacts, "
            "and capital flows. For M&A: deal premium → funding mix → dilution/accretion → "
            "EPS/valuation multiple → synergy realization risk."
        )
