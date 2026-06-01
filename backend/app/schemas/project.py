from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ProjectFile(BaseModel):
    filename: str
    size: int

class ProjectSchema(BaseModel):
    project_id: str
    name: str
    status: str
    simulation_requirement: Optional[str] = None
    ontology: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[str] = None
    files: List[ProjectFile] = []
    total_text_length: int = 0
    graph_id: Optional[str] = None
    graph_build_task_id: Optional[str] = None
    graph_build_progress: int = 0
    graph_build_message: Optional[str] = None
    ontology_task_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
