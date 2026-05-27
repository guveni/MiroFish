from typing import List, Tuple, Dict, Any, Optional

class BaseAnalyticalModule:
    """Base interface for domain-specific analytical modules."""
    
    def get_vague_phrases(self) -> List[str]:
        """Returns list of domain-specific vague phrases to penalize."""
        return []
        
    def check_numerical_sanity(self, text: str, sources_metadata: List[Tuple[int, str, str]]) -> List[str]:
        """Returns list of domain-specific numerical anomalies found in text."""
        return []
        
    def check_quantitative_grounding(self, text: str, sources_metadata: List[Tuple[int, str, str]]) -> List[str]:
        """Returns list of domain-specific missing quantitative or grounding requirements."""
        return []
        
    def get_causal_chain_hints(self) -> str:
        """Returns domain-specific causal chain hints to guide the planner/composer."""
        return ""
