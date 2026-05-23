"""
Graph-related API routes.
Uses project context with server-side persisted state.
"""

import os
import traceback
import threading
from flask import request, jsonify

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services import _backend
from ..services.research_query_generator import ResearchQueryGenerator
from ..services.run_checkpoint_store import (
    checkpoint_project_stage,
    load_project_stage_checkpoint,
)
from ..services.gemini_web_search import (
    EMPTY_GEMINI_WEB_SEARCH_RESULTS,
    GEMINI_WEB_SEARCH_MODEL_NOT_SET,
    GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED,
    gemini_web_search_configured,
    resolve_grounding_model,
    search_queries_to_corpus,
)
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..utils.locale import t, get_locale, set_locale
from ..utils.pipeline_retry import run_pipeline_step
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus

logger = get_logger('mirofish.api')


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

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    Get project details.
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": t('api.projectNotFound', id=project_id)
        }), 404

    data = project.to_dict()
    if project.status == ProjectStatus.GRAPH_BUILDING and project.graph_build_task_id:
        live_task = TaskManager().get_task(project.graph_build_task_id)
        data["graph_build_task_stale"] = live_task is None
    else:
        data["graph_build_task_stale"] = False

    return jsonify({
        "success": True,
        "data": data
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    List all projects.
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    Delete a project.
    """
    success = ProjectManager.delete_project(project_id)
    
    if not success:
        return jsonify({
            "success": False,
            "error": t('api.projectDeleteFailed', id=project_id)
        }), 404

    return jsonify({
        "success": True,
        "message": t('api.projectDeleted', id=project_id)
    })


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    Reset project state for rebuilding the graph.
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": t('api.projectNotFound', id=project_id)
        }), 404

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
    
    return jsonify({
        "success": True,
        "message": t('api.projectReset', id=project_id),
        "data": project.to_dict()
    })


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


@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    Upload files and optionally use Gemini web search grounding, then generate ontology.

    Returns immediately with task_id; poll GET /api/graph/task/<task_id> for progress.
    On completion, task.result contains the same payload as the former synchronous response.
    """
    project = None
    created_project = False
    try:
        logger.info("=== Starting ontology generation (async) ===")

        simulation_requirement = request.form.get('simulation_requirement', '')
        raw_project_name = request.form.get('project_name')
        project_name = raw_project_name or 'Unnamed Project'
        additional_context = request.form.get('additional_context', '')
        use_vertex_search = _form_bool(request.form.get('use_vertex_search'))
        requested_project_id = (
            request.form.get('project_id')
            or request.args.get('project_id')
            or ''
        ).strip()

        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationRequirement')
            }), 400

        if use_vertex_search:
            if not gemini_web_search_configured():
                return jsonify({
                    "success": False,
                    "error": t('api.geminiWebSearchConfigMissing')
                }), 400
            try:
                resolve_grounding_model()
            except ValueError as e:
                if str(e) != GEMINI_WEB_SEARCH_MODEL_NOT_SET:
                    raise
                return jsonify({
                    "success": False,
                    "error": t('api.geminiWebSearchModelMissing')
                }), 400

        uploaded_files = request.files.getlist('files') or []
        has_file_upload = any(f and f.filename for f in uploaded_files)

        checkpoint = None
        if Config.RESUME_FROM_CHECKPOINT and requested_project_id:
            checkpoint = load_project_stage_checkpoint(
                requested_project_id,
                "ontology_generated",
            )

        # Allow retry using stored files when an existing project_id is provided
        if not has_file_upload and not use_vertex_search and not checkpoint and not requested_project_id:
            return jsonify({
                "success": False,
                "error": t('api.requireDocOrVertexSearch')
            }), 400

        existing_project = None
        if requested_project_id:
            existing_project = ProjectManager.get_project(requested_project_id)
            if not existing_project:
                return jsonify({
                    "success": False,
                    "error": t('api.projectNotFound', id=requested_project_id)
                }), 404

        task_manager = TaskManager()
        task_id = task_manager.create_task(
            "ontology_generate",
            metadata={"project_name": project_name},
        )
        task_manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            progress=2,
            message=t('progress.ontologyValidating'),
        )

        if requested_project_id:
            project = existing_project
            project.name = raw_project_name or project.name
        else:
            project = ProjectManager.create_project(name=project_name)
            created_project = True
        project.simulation_requirement = simulation_requirement
        project.ontology_task_id = task_id
        ProjectManager.save_project(project)
        logger.info("Created project: %s (task_id=%s)", project.project_id, task_id)

        if checkpoint:
            _resume_ontology_from_checkpoint(
                project,
                checkpoint["payload"],
                task_id=task_id,
            )
            return jsonify({
                "success": True,
                "data": {
                    "project_id": project.project_id,
                    "task_id": task_id,
                    "message": t('api.ontologyTaskStarted', taskId=task_id),
                    "resumed_from_checkpoint": True,
                },
            })

        saved_file_count = 0
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                file_info = ProjectManager.save_file_to_project(
                    project.project_id,
                    file,
                    file.filename,
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"],
                })
                saved_file_count += 1

        if saved_file_count:
            task_manager.update_task(
                task_id,
                progress=8,
                message=t('progress.ontologySavingFiles', count=saved_file_count),
            )
            ProjectManager.save_project(project)

        current_locale = get_locale()

        def ontology_task():
            set_locale(current_locale)
            worker_logger = get_logger('mirofish.ontology')
            local_project = ProjectManager.get_project(project.project_id)
            if not local_project:
                task_manager.fail_task(task_id, f"Project not found: {project.project_id}")
                return

            try:
                document_texts = []
                all_text = ""
                search_corpus = None
                search_metadata = None

                task_manager.update_task(
                    task_id,
                    progress=12,
                    message=t('progress.ontologyExtractingText', chars=0),
                )
                file_paths = ProjectManager.get_project_files(local_project.project_id)
                for idx, path in enumerate(file_paths):
                    label = (
                        local_project.files[idx]["filename"]
                        if idx < len(local_project.files)
                        else os.path.basename(path)
                    )
                    text = FileParser.extract_text(path)
                    text = TextProcessor.preprocess_text(text)
                    document_texts.append(text)
                    all_text += f"\n\n=== {label} ===\n{text}"

                if use_vertex_search:
                    task_manager.update_task(
                        task_id,
                        progress=25,
                        message=t('progress.ontologyWebSearch', count=0),
                    )
                    try:
                        rqg = ResearchQueryGenerator()
                        queries = rqg.generate_queries(
                            simulation_requirement,
                            additional_context if additional_context else None,
                        )
                        task_manager.update_task(
                            task_id,
                            progress=30,
                            message=t('progress.ontologyWebSearch', count=len(queries)),
                        )
                        search_corpus, search_metadata = search_queries_to_corpus(
                            queries,
                            simulation_requirement=simulation_requirement,
                        )
                    except ValueError as search_err:
                        code = str(search_err)
                        if code == EMPTY_GEMINI_WEB_SEARCH_RESULTS:
                            raise ValueError(t('api.geminiWebSearchEmptyResults')) from search_err
                        if code == GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED:
                            raise ValueError(t('api.geminiWebSearchConfigMissing')) from search_err
                        if code == GEMINI_WEB_SEARCH_MODEL_NOT_SET:
                            raise ValueError(t('api.geminiWebSearchModelMissing')) from search_err
                        raise
                    if search_corpus:
                        all_text += f"\n\n=== gemini_web_search ===\n{search_corpus}"

                if not document_texts and not (use_vertex_search and search_corpus):
                    raise ValueError(t('api.noDocProcessed'))

                if not all_text.strip():
                    raise ValueError(t('api.geminiWebSearchEmptyResults'))

                char_count = len(all_text)
                local_project.total_text_length = char_count
                ProjectManager.save_extracted_text(local_project.project_id, all_text)
                ProjectManager.save_project(local_project)
                worker_logger.info(
                    "[%s] Text extraction complete: %d characters",
                    task_id,
                    char_count,
                )
                task_manager.update_task(
                    task_id,
                    progress=40,
                    message=t('progress.ontologyExtractingText', chars=char_count),
                )

                task_manager.update_task(
                    task_id,
                    progress=50,
                    message=t('progress.ontologyCallingLlm'),
                )
                worker_logger.info("[%s] Calling LLM to generate ontology...", task_id)
                generator = OntologyGenerator()

                def llm_progress(msg: str, pct: int) -> None:
                    task_manager.update_task(
                        task_id,
                        progress=max(50, min(88, pct)),
                        message=msg,
                    )

                ontology = generator.generate(
                    document_texts=document_texts,
                    simulation_requirement=simulation_requirement,
                    additional_context=additional_context if additional_context else None,
                    web_search_text=search_corpus,
                    progress_callback=llm_progress,
                )

                task_manager.update_task(
                    task_id,
                    progress=92,
                    message=t('progress.ontologyProcessingResult'),
                )

                entity_count = len(ontology.get("entity_types", []))
                edge_count = len(ontology.get("edge_types", []))
                local_project.ontology = {
                    "entity_types": ontology.get("entity_types", []),
                    "edge_types": ontology.get("edge_types", []),
                }
                local_project.analysis_summary = ontology.get("analysis_summary", "")
                local_project.status = ProjectStatus.ONTOLOGY_GENERATED
                local_project.gemini_grounding_metadata = (
                    search_metadata if use_vertex_search else None
                )
                local_project.ontology_task_id = None
                local_project.error = None

                task_manager.update_task(
                    task_id,
                    progress=96,
                    message=t('progress.ontologySaving'),
                )
                ProjectManager.save_project(local_project)
                worker_logger.info(
                    "[%s] Ontology generated: %d entity types, %d relationship types",
                    task_id,
                    entity_count,
                    edge_count,
                )

                try:
                    checkpoint_project_stage(
                        local_project.project_id,
                        "ontology_generated",
                        {
                            "project_id": local_project.project_id,
                            "ontology_entity_types": entity_count,
                            "ontology_edge_types": edge_count,
                            "ontology": local_project.ontology,
                            "analysis_summary": local_project.analysis_summary,
                            "files": local_project.files,
                            "total_text_length": local_project.total_text_length,
                            "use_vertex_search": use_vertex_search,
                            "gemini_grounding_metadata": (
                                search_metadata if use_vertex_search else None
                            ),
                        },
                    )
                except Exception as cp_err:
                    worker_logger.warning(
                        "run checkpoint ontology_generated skipped: %s", cp_err
                    )

                result_payload = _ontology_result_payload(
                    local_project,
                    search_metadata=search_metadata,
                    use_vertex_search=use_vertex_search,
                )
                task_manager.update_task(
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
                worker_logger.info(
                    "=== Ontology generation complete === project_id=%s",
                    local_project.project_id,
                )

            except ValueError as e:
                err_msg = str(e)
                worker_logger.error("[%s] Ontology generation failed: %s", task_id, err_msg)
                local_project.status = ProjectStatus.FAILED
                local_project.error = err_msg
                local_project.ontology_task_id = None
                ProjectManager.save_project(local_project)
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=t('progress.ontologyFailed', error=err_msg),
                    error=err_msg,
                )
            except Exception as e:
                worker_logger.exception("[%s] Ontology generation failed", task_id)
                err_msg = str(e)
                local_project.status = ProjectStatus.FAILED
                local_project.error = err_msg
                local_project.ontology_task_id = None
                ProjectManager.save_project(local_project)
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=t('progress.ontologyFailed', error=err_msg),
                    error=traceback.format_exc(),
                )

        thread = threading.Thread(target=ontology_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "task_id": task_id,
                "message": t('api.ontologyTaskStarted', taskId=task_id),
            },
        })

    except Exception as e:
        logger.exception("Ontology generation setup failed: %s", e)
        if project and created_project:
            try:
                ProjectManager.delete_project(project.project_id)
            except Exception:
                pass
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== API 2: Build Graph ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """
    Build a graph from project_id.
    
    Request (JSON):
        {
            "project_id": "proj_xxxx",
            "graph_name": "Graph name",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "Graph build task has started"
            }
        }
    """
    try:
        logger.info("=== Starting graph build ===")
        
        # Validate configuration.
        backend_ok, backend_error_key = _backend.is_available()
        if not backend_ok:
            error = t(backend_error_key or 'api.graphBackendUnavailable')
            logger.error("Graph backend unavailable: %s", error)
            return jsonify({
                "success": False,
                "error": t('api.configError', details=error)
            }), 500
        
        # Parse request.
        data = request.get_json() or {}
        project_id = data.get('project_id')
        logger.debug("Request parameter: project_id=%s", project_id)
        
        if not project_id:
            return jsonify({
                "success": False,
                "error": t('api.requireProjectId')
            }), 400
        
        # Load project.
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=project_id)
            }), 404

        # Check project status.
        force = data.get('force', False)
        
        if project.status == ProjectStatus.CREATED:
            return jsonify({
                "success": False,
                "error": t('api.ontologyNotGenerated')
            }), 400
        
        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify({
                "success": False,
                "error": t('api.graphBuilding'),
                "task_id": project.graph_build_task_id
            }), 400
        
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
                    return jsonify({
                        "success": True,
                        "data": {
                            "project_id": project_id,
                            "task_id": task_id,
                            "message": t('api.graphBuildStarted', taskId=task_id),
                            "resumed_from_checkpoint": True,
                        }
                    })
        
        # Load extracted text.
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify({
                "success": False,
                "error": t('api.textNotFound')
            }), 400
        
        # Load ontology.
        ontology = project.ontology
        if not ontology:
            return jsonify({
                "success": False,
                "error": t('api.ontologyNotFound')
            }), 400
        
        # Create async task.
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Build graph: {graph_name}")
        logger.info("Created graph build task: task_id=%s, project_id=%s", task_id, project_id)
        
        # Update project state.
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        _clear_build_progress(project)
        ProjectManager.save_project(project)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Start background task.
        def build_task():
            set_locale(current_locale)
            build_logger = get_logger('mirofish.build')

            def update_build_task(*, progress=None, message=None, **kwargs):
                task_manager.update_task(
                    task_id,
                    progress=progress,
                    message=message,
                    **kwargs,
                )
                if progress is not None or message:
                    _persist_build_progress(
                        project,
                        progress if progress is not None else int(project.graph_build_progress or 0),
                        message or project.graph_build_message or "",
                    )

            try:
                build_logger.info("[%s] Starting graph build...", task_id)
                update_build_task(
                    status=TaskStatus.PROCESSING,
                    message=t('progress.initGraphService'),
                )
                
                # Create graph build service.
                builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
                
                # Split text into chunks.
                update_build_task(
                    message=t('progress.textChunking'),
                    progress=5,
                )
                chunks = TextProcessor.split_text(
                    text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                total_chunks = len(chunks)
                
                # Create graph.
                update_build_task(
                    message=t('progress.creatingZepGraph'),
                    progress=10,
                )
                graph_id = run_pipeline_step(
                    "zep_create_graph",
                    lambda gn=graph_name: builder.create_graph(name=gn),
                )

                project.graph_id = graph_id
                ProjectManager.save_project(project)

                update_build_task(
                    message=t('progress.settingOntology'),
                    progress=15,
                )
                run_pipeline_step(
                    "zep_set_ontology",
                    lambda gid=graph_id: builder.set_ontology(gid, ontology),
                )
                
                # Add text. progress_callback signature is (msg, progress_ratio).
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    update_build_task(message=msg, progress=progress)
                
                update_build_task(
                    message=t('progress.addingChunks', count=total_chunks),
                    progress=15,
                )
                
                episode_uuids = builder.add_text_batches(
                    graph_id,
                    chunks,
                    batch_size=Config.GRAPHITI_BATCH_SIZE,
                    progress_callback=add_progress_callback,
                )
                
                # Wait for Zep processing by checking each episode's processed state.
                update_build_task(
                    message=t('progress.waitingZepProcess'),
                    progress=55,
                )
                
                def wait_progress_callback(msg, progress_ratio):
                    progress = 55 + int(progress_ratio * 35)  # 55% - 90%
                    update_build_task(message=msg, progress=progress)
                
                run_pipeline_step(
                    "zep_wait_for_episodes",
                    lambda uuids=episode_uuids: builder._wait_for_episodes(
                        uuids,
                        wait_progress_callback,
                    ),
                )
                
                # Fetch graph data.
                update_build_task(
                    message=t('progress.fetchingGraphData'),
                    progress=95,
                )
                graph_data = run_pipeline_step(
                    "zep_get_graph_data",
                    lambda gid=graph_id: builder.get_graph_data(gid),
                )
                
                # Update project state.
                project.status = ProjectStatus.GRAPH_COMPLETED
                project.graph_build_task_id = None
                _persist_build_progress(
                    project,
                    100,
                    t('progress.graphBuildComplete'),
                )
                ProjectManager.save_project(project)
                
                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                build_logger.info(
                    "[%s] Graph build complete: graph_id=%s, nodes=%d, edges=%d",
                    task_id,
                    graph_id,
                    node_count,
                    edge_count,
                )
                
                # Complete task.
                update_build_task(
                    status=TaskStatus.COMPLETED,
                    message=t('progress.graphBuildComplete'),
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks
                    }
                )
                try:
                    checkpoint_project_stage(
                        project_id,
                        "graph_completed",
                        {
                            "project_id": project_id,
                            "graph_id": graph_id,
                            "task_id": task_id,
                            "node_count": node_count,
                            "edge_count": edge_count,
                            "chunk_count": total_chunks,
                        },
                    )
                except Exception as cp_err:
                    build_logger.warning("run checkpoint graph_completed skipped: %s", cp_err)
                
            except Exception as e:
                # Mark project as failed.
                build_logger.error("[%s] Graph build failed: %s", task_id, str(e))
                build_logger.debug(traceback.format_exc())
                
                project.status = ProjectStatus.FAILED
                project.graph_build_task_id = None
                project.error = str(e)
                _persist_build_progress(
                    project,
                    int(project.graph_build_progress or 0),
                    t('progress.buildFailed', error=str(e)),
                )
                ProjectManager.save_project(project)
                
                update_build_task(
                    status=TaskStatus.FAILED,
                    message=t('progress.buildFailed', error=str(e)),
                    error=traceback.format_exc(),
                )
                try:
                    checkpoint_project_stage(
                        project_id,
                        "graph_failed",
                        {
                            "project_id": project_id,
                            "task_id": task_id,
                            "graph_id": getattr(project, "graph_id", None),
                        },
                        error=str(e),
                    )
                except Exception as cp_err:
                    build_logger.warning("run checkpoint graph_failed skipped: %s", cp_err)
        
        # Start background thread.
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": t('api.graphBuildStarted', taskId=task_id)
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Task Query API ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Query task status.
    """
    task = TaskManager().get_task(task_id)
    
    if not task:
        return jsonify({
            "success": False,
            "error": t('api.taskNotFound', id=task_id)
        }), 404
    
    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all tasks.
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== Graph Data API ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    Get graph data, including nodes and edges.
    """
    try:
        backend_ok, backend_error_key = _backend.is_available()
        if not backend_ok:
            return jsonify({
                "success": False,
                "error": t(backend_error_key or 'api.graphBackendUnavailable')
            }), 500
        
        builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
        graph_data = builder.get_graph_data(graph_id)
        
        return jsonify({
            "success": True,
            "data": graph_data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    Delete a knowledge graph.
    """
    try:
        backend_ok, backend_error_key = _backend.is_available()
        if not backend_ok:
            return jsonify({
                "success": False,
                "error": t(backend_error_key or 'api.graphBackendUnavailable')
            }), 500
        
        builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
        builder.delete_graph(graph_id)
        
        return jsonify({
            "success": True,
            "message": t('api.graphDeleted', id=graph_id)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
