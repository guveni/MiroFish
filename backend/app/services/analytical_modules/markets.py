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
        
    def get_causal_chain_hints(self) -> str:
        return "Market Specific: Link price movements directly to supply/demand shocks, P/E impacts, or capital outflows."
