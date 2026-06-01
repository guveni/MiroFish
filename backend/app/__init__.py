"""
MiroFish Backend - FastAPI Application Factory
"""

import os
import warnings

# Suppress multiprocessing resource_tracker warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app() -> FastAPI:
    """FastAPI application factory function."""
    app = FastAPI(title="MiroFish API", version="0.1.0")
    
    # Initialize database
    from .database import db
    db.init_app(Config.SQLALCHEMY_DATABASE_URI)
    
    # Import models to register and automatically create tables
    from .models.user import User
    from .models.project import Project
    from .models.task import Task
    db.create_all()
    
    # Setup logger
    logger = setup_logger('mirofish')
    
    # Print startup message
    logger.info("=" * 50)
    logger.info("MiroFish Backend (FastAPI) Starting...")
    logger.info("=" * 50)
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register simulation clean-up on server shutdown
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    logger.info("Registered simulation cleanup callback.")
    
    # Request logging and context middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        req_logger = get_logger('mirofish.request')
        req_logger.debug(f"Request: {request.method} {request.url.path}")
        
        from .utils.locale import request_var
        token = request_var.set(request)
        try:
            response = await call_next(request)
            req_logger.debug(f"Response: {response.status_code}")
            return response
        finally:
            request_var.reset(token)
    
    # DB session cleanup middleware to prevent connection leaks
    @app.middleware("http")
    async def db_session_cleanup(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        finally:
            from .database import db
            db.session.remove()
            
    # Centralized global exception handlers
    import traceback
    from fastapi.exceptions import RequestValidationError
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.detail
                }
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        def _json_safe(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            return value

        errors_list = []
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error.get("loc", []))
            msg = error.get("msg", "Validation error")
            errors_list.append(f"{loc}: {msg}")
        error_msg = "; ".join(errors_list)
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": error_msg,
                "details": _json_safe(exc.errors())
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Global exception handler caught: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc()
            }
        )
    
    # Register API routers
    from .api import graph_router, simulation_router, report_router
    app.include_router(graph_router, prefix="/api/graph")
    app.include_router(simulation_router, prefix="/api/simulation")
    app.include_router(report_router, prefix="/api/report")
    
    # Register Inngest FastAPI adapter
    import inngest.fast_api
    from .inngest_client import inngest_client
    from .tasks.inngest_tasks import generate_ontology_task, build_graph_task, generate_report_task, run_simulation_task
    inngest.fast_api.serve(
        app,
        inngest_client,
        [generate_ontology_task, build_graph_task, generate_report_task, run_simulation_task]
    )
    
    # Health check
    @app.get("/health")
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
        
    # Serve unified Vue 3 static assets if compiled dist exists
    frontend_dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/dist"))
    if os.path.exists(frontend_dist_dir):
        assets_dir = os.path.join(frontend_dist_dir, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
            logger.info(f"Mounted Vue 3 static assets from {assets_dir}")
            
        # Catch-all route to serve the single page app (SPA) HTML5 history
        @app.get("/{catchall:path}")
        async def serve_spa(catchall: str):
            if catchall.startswith("api/") or catchall.startswith("health") or catchall.startswith("assets/"):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            index_path = os.path.join(frontend_dist_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return JSONResponse(status_code=404, content={"detail": "Index file not found"})
    else:
        logger.info("Vue 3 frontend dist folder not found. Unified static serving disabled (development mode).")
        
    logger.info("MiroFish Backend (FastAPI) Initialization Complete.")
    return app
