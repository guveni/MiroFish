"""
Graph-related API routes (FastAPI native).
Uses project context with server-side persisted state.
"""

import os
import traceback
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, Query, status
from fastapi.responses import JSONResponse

from starlette.concurrency import run_in_threadpool
from ...utils.auth import get_current_user
from ...config import Config
from ...services.graph_builder import GraphBuilderService
from ...services import _backend
from ...services.run_checkpoint_store import (
    load_project_stage_checkpoint,
)
from ...services.web_search import (
    GEMINI_WEB_SEARCH_MODEL_NOT_SET,
    gemini_web_search_configured,
    resolve_grounding_model,
)
from ...utils.logger import get_logger
from ...utils.locale import t, get_locale
from ...models.task import TaskManager, TaskStatus
from ...models.project import Project, ProjectManager, ProjectStatus
from ...models.user import User
from ...schemas.base import StandardResponse
from ...schemas.project import ProjectSchema

logger = get_logger('mirofish.api')

router = APIRouter()


def allowed_file(filename: str) -> bool:
    """Return whether the file extension is allowed."""
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


def _form_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _persist_build_progress(project, progress: int, message: str) -> None:
    """Mirror in-memory task progress onto the project for UI resume and restarts."""
    project.graph_build_progress = max(int(project.graph_build_progress or 0), int(progress))
    if message:
        project.graph_build_message = message
    ProjectManager.save_project(project)


def _clear_build_progress(project) -> None:
    project.graph_build_progress = 0
    project.graph_build_message = ""


# ============== Project Management ==============

@router.get('/project/{project_id}', response_model=StandardResponse[ProjectSchema])
def get_project(project_id: str, current_user: User = Depends(get_current_user)):
    """Get project details."""
    project = ProjectManager.get_project(project_id)
    
    if not project or (project.user_id and project.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=t('api.projectNotFound', id=project_id)
        )

    data = project.to_dict()
    if project.status == ProjectStatus.GRAPH_BUILDING and project.graph_build_task_id:
        live_task = TaskManager().get_task(project.graph_build_task_id)
        data["graph_build_task_stale"] = live_task is None
    else:
        data["graph_build_task_stale"] = False

    return {
        "success": True,
        "data": data
    }


@router.get('/project/list')
def list_projects(limit: int = 50, current_user: User = Depends(get_current_user)):
    """List all projects."""
    projects = ProjectManager.list_projects(user_id=current_user.id, limit=limit)
    return {
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    }


@router.delete('/project/{project_id}')
def delete_project(project_id: str, current_user: User = Depends(get_current_user)):
    """Delete a project."""
    project = ProjectManager.get_project(project_id)
    if not project or (project.user_id and project.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=t('api.projectNotFound', id=project_id)
        )
        
    success = ProjectManager.delete_project(project_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=t('api.projectDeleteFailed', id=project_id)
        )

    return {
        "success": True,
        "message": t('api.projectDeleted', id=project_id)
    }


@router.post('/project/{project_id}/reset')
def reset_project(project_id: str, current_user: User = Depends(get_current_user)):
    """Reset project state for rebuilding the graph."""
    project = ProjectManager.get_project(project_id)
    
    if not project or (project.user_id and project.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=t('api.projectNotFound', id=project_id)
        )

    # Reset to the ontology-generated state.
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED
    
    project.graph_id = None
    project.graph_build_task_id = None
    _clear_build_progress(project)
    project.ontology_task_id = None
    project.error = None
    ProjectManager.save_project(project)
    
    return {
        "success": True,
        "message": t('api.projectReset', id=project_id),
        "data": project.to_dict()
    }


# ============== API 1: Upload Files And Generate Ontology ==============

def _ontology_result_payload(project, *, search_metadata=None, use_vertex_search=False):
    """Build API payload after ontology generation completes."""
    payload = {
        "project_id": project.project_id,
        "project_name": project.name,
        "ontology": project.ontology,
        "analysis_summary": project.analysis_summary,
        "files": project.files,
        "total_text_length": project.total_text_length,
        "gemini_grounding_metadata": project.gemini_grounding_metadata,
    }
    if use_vertex_search and search_metadata is not None:
        payload["search_metadata"] = search_metadata
    return payload


def _resume_ontology_from_checkpoint(project, payload: dict, *, task_id: str):
    """Hydrate a project from an ontology_generated checkpoint."""
    project.ontology = payload.get("ontology")
    project.analysis_summary = payload.get("analysis_summary", "")
    project.files = payload.get("files", project.files)
    project.total_text_length = payload.get("total_text_length", project.total_text_length)
    project.gemini_grounding_metadata = payload.get("gemini_grounding_metadata")
    project.status = ProjectStatus.ONTOLOGY_GENERATED
    project.ontology_task_id = None
    project.error = None
    ProjectManager.save_project(project)

    use_vertex_search = bool(payload.get("use_vertex_search"))
    search_metadata = payload.get("gemini_grounding_metadata")
    result_payload = _ontology_result_payload(
        project,
        search_metadata=search_metadata,
        use_vertex_search=use_vertex_search,
    )
    result_payload["resumed_from_checkpoint"] = True

    entity_count = len((project.ontology or {}).get("entity_types", []))
    edge_count = len((project.ontology or {}).get("edge_types", []))
    TaskManager().update_task(
        task_id,
        status=TaskStatus.COMPLETED,
        progress=100,
        message=t(
            'progress.ontologyComplete',
            entities=entity_count,
            edges=edge_count,
        ),
        result=result_payload,
    )
    return result_payload


@router.post('/ontology/generate')
async def generate_ontology(
    simulation_requirement: str = Form(""),
    project_name: Optional[str] = Form(None),
    additional_context: str = Form(""),
    use_vertex_search: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    force: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    current_user: User = Depends(get_current_user)
):
    """
    Upload files and optionally use Gemini web search grounding, then generate ontology.
    Returns immediately with task_id; poll GET /api/graph/task/<task_id> for progress.
    """
    project = None
    created_project = False
    try:
        logger.info("=== Starting ontology generation (async) ===")

        project_name_str = project_name or 'Unnamed Project'
        use_vertex_search_bool = _form_bool(use_vertex_search)
        force_bool = _form_bool(force)
        requested_project_id = (project_id or '').strip()

        if not simulation_requirement:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationRequirement')
                }
            )

        if use_vertex_search_bool:
            if not gemini_web_search_configured():
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.geminiWebSearchConfigMissing')
                    }
                )
            try:
                resolve_grounding_model()
            except ValueError as e:
                if str(e) != GEMINI_WEB_SEARCH_MODEL_NOT_SET:
                    raise
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.geminiWebSearchModelMissing')
                    }
                )

        uploaded_files = files or []
        has_file_upload = any(f and f.filename for f in uploaded_files)
        logger.info(
            "Ontology upload: %d file part(s) received, use_vertex_search=%s",
            len(uploaded_files),
            use_vertex_search_bool,
        )

        checkpoint = None
        if Config.RESUME_FROM_CHECKPOINT and requested_project_id and not force_bool:
            checkpoint = await run_in_threadpool(
                load_project_stage_checkpoint,
                requested_project_id,
                "ontology_generated",
            )

        # Require at least one seed source: uploaded files, web search, or simulation requirement text.
        has_requirement_seed = bool(simulation_requirement.strip())
        if (
            not has_file_upload
            and not use_vertex_search_bool
            and not checkpoint
            and not requested_project_id
            and not has_requirement_seed
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireDocOrVertexSearch')
                }
            )

        existing_project = None
        if requested_project_id:
            existing_project = await run_in_threadpool(ProjectManager.get_project, requested_project_id)
            if not existing_project or (existing_project.user_id and existing_project.user_id != current_user.id):
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "error": t('api.projectNotFound', id=requested_project_id)
                    }
                )

        task_manager = TaskManager()
        task_id = await run_in_threadpool(
            task_manager.create_task,
            "ontology_generate",
            metadata={"project_name": project_name_str},
            user_id=current_user.id,
        )
        await run_in_threadpool(
            task_manager.update_task,
            task_id,
            status=TaskStatus.PROCESSING,
            progress=2,
            message=t('progress.ontologyValidating'),
        )

        if requested_project_id:
            project = existing_project
            project.name = project_name or project.name
            project.status = ProjectStatus.CREATED.value
            project.error = None
        else:
            project = await run_in_threadpool(ProjectManager.create_project, name=project_name_str, user_id=current_user.id)
            created_project = True
        project.simulation_requirement = simulation_requirement
        project.ontology_task_id = task_id
        await run_in_threadpool(ProjectManager.save_project, project)
        logger.info("Created project: %s (task_id=%s)", project.project_id, task_id)

        if checkpoint:
            await run_in_threadpool(
                _resume_ontology_from_checkpoint,
                project,
                checkpoint["payload"],
                task_id=task_id,
            )
            return {
                "success": True,
                "data": {
                    "project_id": project.project_id,
                    "task_id": task_id,
                    "message": t('api.ontologyTaskStarted', taskId=task_id),
                    "resumed_from_checkpoint": True,
                },
            }

        saved_file_count = 0
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                file_info = await run_in_threadpool(
                    ProjectManager.save_file_to_project,
                    project.project_id,
                    file,
                    file.filename,
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"],
                })
                saved_file_count += 1

        if has_file_upload and saved_file_count == 0:
            logger.warning(
                "Ontology upload included file parts but none were saved "
                "(allowed: %s)",
                ", ".join(sorted(Config.ALLOWED_EXTENSIONS)),
            )

        if saved_file_count:
            await run_in_threadpool(
                task_manager.update_task,
                task_id,
                progress=8,
                message=t('progress.ontologySavingFiles', count=saved_file_count),
            )
            await run_in_threadpool(ProjectManager.save_project, project)

        stored_file_paths = await run_in_threadpool(
            ProjectManager.get_project_files,
            project.project_id,
        )
        if (
            not use_vertex_search_bool
            and not checkpoint
            and saved_file_count == 0
            and not stored_file_paths
            and not has_requirement_seed
        ):
            await run_in_threadpool(
                task_manager.fail_task,
                task_id,
                t('api.noStoredProjectFiles'),
            )
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.noStoredProjectFiles'),
                    "task_id": task_id,
                },
            )

        current_locale = get_locale()

        # Send event to Inngest for background execution
        from ...inngest_client import inngest_client
        import inngest
        await run_in_threadpool(
            inngest_client.send_sync,
            inngest.Event(
                name="ontology/generate",
                data={
                    "project_id": project.project_id,
                    "task_id": task_id,
                    "simulation_requirement": simulation_requirement,
                    "additional_context": additional_context,
                    "use_vertex_search": use_vertex_search_bool,
                    "locale": current_locale,
                }
            )
        )

        return {
            "success": True,
            "data": {
                "project_id": project.project_id,
                "task_id": task_id,
                "message": t('api.ontologyTaskStarted', taskId=task_id),
            },
        }

    except Exception as e:
        logger.exception("Ontology generation setup failed: %s", e)
        if project and created_project:
            try:
                await run_in_threadpool(ProjectManager.delete_project, project.project_id)
            except Exception:
                pass
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# ============== API 2: Build Graph ==============

@router.post('/build')
def build_graph(data: dict, current_user: User = Depends(get_current_user)):
    """Build a graph from project_id."""
    try:
        logger.info("=== Starting graph build ===")
        
        # Validate configuration.
        backend_ok, backend_error_key = _backend.is_available()
        if not backend_ok:
            error = t(backend_error_key or 'api.graphBackendUnavailable')
            logger.error("Graph backend unavailable: %s", error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=t('api.configError', details=error)
            )
        
        project_id = data.get('project_id')
        logger.debug("Request parameter: project_id=%s", project_id)
        
        if not project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=t('api.requireProjectId')
            )
        
        # Load project.
        project = ProjectManager.get_project(project_id)
        if not project or (project.user_id and project.user_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=t('api.projectNotFound', id=project_id)
            )

        # Check project status.
        force = data.get('force', False)
        
        if project.status == ProjectStatus.CREATED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=t('api.ontologyNotGenerated')
            )
        
        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.graphBuilding'),
                    "task_id": project.graph_build_task_id
                }
            )
        
        # Reset state for forced rebuilds.
        if force and project.status in [ProjectStatus.GRAPH_BUILDING, ProjectStatus.FAILED, ProjectStatus.GRAPH_COMPLETED]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            _clear_build_progress(project)
            project.error = None
        
        # Load configuration.
        graph_name = data.get('graph_name', project.name or 'MiroFish Graph')
        chunk_size = data.get('chunk_size', project.chunk_size or Config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = data.get('chunk_overlap', project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP)
        
        # Update project configuration.
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap

        if Config.RESUME_FROM_CHECKPOINT and not force:
            checkpoint = load_project_stage_checkpoint(project_id, "graph_completed")
            if checkpoint:
                payload = checkpoint["payload"]
                graph_id = payload.get("graph_id")
                if graph_id:
                    task_manager = TaskManager()
                    task_id = task_manager.create_task(f"Build graph: {graph_name}")
                    project.graph_id = graph_id
                    project.graph_build_task_id = task_id
                    project.status = ProjectStatus.GRAPH_COMPLETED
                    ProjectManager.save_project(project)
                    result = {
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": payload.get("node_count", 0),
                        "edge_count": payload.get("edge_count", 0),
                        "chunk_count": payload.get("chunk_count", 0),
                        "resumed_from_checkpoint": True,
                    }
                    task_manager.complete_task(task_id, result)
                    return {
                        "success": True,
                        "data": {
                            "project_id": project_id,
                            "task_id": task_id,
                            "message": t('api.graphBuildStarted', taskId=task_id),
                            "resumed_from_checkpoint": True,
                        }
                    }
        
        # Load extracted text.
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=t('api.textNotFound')
            )
        
        # Load ontology.
        ontology = project.ontology
        if not ontology:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=t('api.ontologyNotFound')
            )
        
        # Create async task.
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Build graph: {graph_name}", user_id=current_user.id)
        logger.info("Created graph build task: task_id=%s, project_id=%s", task_id, project_id)
        
        # Update project state.
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        _clear_build_progress(project)
        ProjectManager.save_project(project)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Send event to Inngest for background execution
        from ...inngest_client import inngest_client
        import inngest
        inngest_client.send_sync(
            inngest.Event(
                name="graph/build",
                data={
                    "project_id": project_id,
                    "task_id": task_id,
                    "graph_name": graph_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "locale": current_locale,
                }
            )
        )
        
        return {
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": t('api.graphBuildStarted', taskId=task_id)
            }
        }

    except Exception as e:
        logger.exception("Graph build setup failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# ============== Task Query API ==============

@router.get('/task/{task_id}')
def get_task(task_id: str, current_user: User = Depends(get_current_user)):
    """Query task status."""
    task = TaskManager().get_task(task_id)
    
    if not task or (task.user_id and task.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=t('api.taskNotFound', id=task_id)
        )
    
    return {
        "success": True,
        "data": task.to_dict()
    }


@router.get('/tasks')
def list_tasks(current_user: User = Depends(get_current_user)):
    """List all tasks."""
    tasks = TaskManager().list_tasks(user_id=current_user.id)
    return {
        "success": True,
        "data": tasks,
        "count": len(tasks)
    }


# ============== Graph Data API ==============

@router.get('/data/{graph_id}')
def get_graph_data(graph_id: str, current_user: User = Depends(get_current_user)):
    """Get graph data, including nodes and edges."""
    # Check authorization via project lookup
    project = Project.query.filter_by(graph_id=graph_id).first()
    if not project or (project.user_id and project.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this graph."
        )

    backend_ok, backend_error_key = _backend.is_available()
    if not backend_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=t(backend_error_key or 'api.graphBackendUnavailable')
        )
    
    builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
    graph_data = builder.get_graph_data(graph_id)
    
    return {
        "success": True,
        "data": graph_data
    }


@router.delete('/delete/{graph_id}')
def delete_graph(graph_id: str, current_user: User = Depends(get_current_user)):
    """Delete a knowledge graph."""
    # Check authorization via project lookup
    project = Project.query.filter_by(graph_id=graph_id).first()
    if not project or (project.user_id and project.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this graph."
        )

    backend_ok, backend_error_key = _backend.is_available()
    if not backend_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=t(backend_error_key or 'api.graphBackendUnavailable')
        )
    
    builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
    builder.delete_graph(graph_id)
    
    return {
        "success": True,
        "message": t('api.graphDeleted', id=graph_id)
    }
