"""
API Router Initialization (FastAPI Native)
"""

from .graph.router import router as graph_router
from .simulation.router import router as simulation_router
from .report.router import router as report_router

__all__ = ["graph_router", "simulation_router", "report_router"]
