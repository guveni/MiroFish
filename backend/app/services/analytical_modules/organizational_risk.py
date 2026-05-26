from typing import List, Tuple
from . import BaseAnalyticalModule

class OrganizationalRiskAnalyticalModule(BaseAnalyticalModule):
    """Domain-specific analytical rules for organizational and operational risk."""
    
    def get_vague_phrases(self) -> List[str]:
        return ["organizational challenges", "operational issues", "risk management plans", "improving internal communication"]
        
    def check_numerical_sanity(self, text: str, sources_metadata: List[Tuple[int, str, str]]) -> List[str]:
        anomalies = []
        import re
        severity_matches = re.finditer(r'\bseverity\s*(?:level|score|index)?\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)\b', text, re.IGNORECASE)
        for sm in severity_matches:
            score = float(sm.group(1))
            scale = float(sm.group(2))
            if score > scale:
                anomalies.append(f"Operational risk severity score of {score}/{scale} exceeds the scale bounds.")
        return anomalies
        
    def get_causal_chain_hints(self) -> str:
        return "Operational Specific: Map event directly to business continuity impact, key-person risk, or supply-chain failure modes."
