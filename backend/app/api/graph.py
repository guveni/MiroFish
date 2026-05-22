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

    return jsonify({
        "success": True,
        "data": project.to_dict()
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
    project.error = None
    ProjectManager.save_project(project)
    
    return jsonify({
        "success": True,
        "message": t('api.projectReset', id=project_id),
        "data": project.to_dict()
    })


# ============== API 1: Upload Files And Generate Ontology ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    Upload files and optionally use Gemini web search grounding, then generate ontology.
    
    Request: multipart/form-data
    
    Parameters:
        files: Uploaded files (PDF/MD/TXT); optional when use_vertex_search is enabled.
        simulation_requirement: Required simulation requirement.
        use_vertex_search: Optional Gemini web search grounding toggle.
        project_name: Optional project name.
        additional_context: Optional extra context.
        
    Returns:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {...},
                "files": [...],
                "total_text_length": 12345,
                "search_metadata": {...}
                "gemini_grounding_metadata": {...}
            }
        }
    """
    project = None
    try:
        logger.info("=== Starting ontology generation ===")
        
        # Parse parameters.
        simulation_requirement = request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Unnamed Project')
        additional_context = request.form.get('additional_context', '')
        use_vertex_search = _form_bool(request.form.get('use_vertex_search'))

        logger.debug("Project name: %s", project_name)
        logger.debug("Simulation requirement: %s...", simulation_requirement[:100])
        
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

        if not has_file_upload and not use_vertex_search:
            return jsonify({
                "success": False,
                "error": t('api.requireDocOrVertexSearch')
            }), 400

        search_corpus = None
        search_metadata = None
        if use_vertex_search:
            try:
                rqg = ResearchQueryGenerator()
                queries = rqg.generate_queries(
                    simulation_requirement,
                    additional_context if additional_context else None,
                )
                search_corpus, search_metadata = search_queries_to_corpus(
                    queries,
                    simulation_requirement=simulation_requirement,
                )
            except ValueError as e:
                if str(e) == EMPTY_GEMINI_WEB_SEARCH_RESULTS:
                    return jsonify({
                        "success": False,
                        "error": t('api.geminiWebSearchEmptyResults')
                    }), 400
                if str(e) == GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED:
                    return jsonify({
                        "success": False,
                        "error": t('api.geminiWebSearchConfigMissing')
                    }), 400
                if str(e) == GEMINI_WEB_SEARCH_MODEL_NOT_SET:
                    return jsonify({
                        "success": False,
                        "error": t('api.geminiWebSearchModelMissing')
                    }), 400
                raise

        # Create project.
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = simulation_requirement
        logger.info("Created project: %s", project.project_id)
        
        # Save files and extract text.
        document_texts = []
        all_text = ""
        
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # Save file to the project directory.
                file_info = ProjectManager.save_file_to_project(
                    project.project_id, 
                    file, 
                    file.filename
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"]
                })
                
                # Extract text.
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                document_texts.append(text)
                all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"

        if use_vertex_search and search_corpus:
            all_text += f"\n\n=== gemini_web_search ===\n{search_corpus}"

        if not document_texts and not (use_vertex_search and search_corpus):
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": t('api.noDocProcessed')
            }), 400
        
        if not all_text.strip():
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": t('api.geminiWebSearchEmptyResults')
            }), 400

        # Save extracted text.
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        logger.info("Text extraction complete: %d characters", len(all_text))
        
        # Generate ontology.
        logger.info("Calling LLM to generate ontology...")
        generator = OntologyGenerator()
        ontology = generator.generate(
            document_texts=document_texts,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context if additional_context else None,
            web_search_text=search_corpus,
        )
        
        # Save ontology to the project.
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(
            "Ontology generated: %d entity types, %d relationship types",
            entity_count,
            edge_count,
        )
        
        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", [])
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        if use_vertex_search and search_metadata is not None:
            project.gemini_grounding_metadata = search_metadata
        else:
            project.gemini_grounding_metadata = None

        ProjectManager.save_project(project)
        logger.info("=== Ontology generation complete === project_id=%s", project.project_id)

        try:
            checkpoint_project_stage(
                project.project_id,
                "ontology_generated",
                {
                    "project_id": project.project_id,
                    "ontology_entity_types": entity_count,
                    "ontology_edge_types": edge_count,
                    "ontology": project.ontology,
                    "analysis_summary": project.analysis_summary,
                    "files": project.files,
                    "total_text_length": project.total_text_length,
                    "use_vertex_search": use_vertex_search,
                    "gemini_grounding_metadata": (
                        search_metadata if use_vertex_search else None
                    ),
                },
            )
        except Exception as cp_err:
            logger.warning("run checkpoint ontology_generated skipped: %s", cp_err)

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
        
        return jsonify({
            "success": True,
            "data": payload
        })
        
    except Exception as e:
        logger.exception("Ontology generation failed: %s", e)
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
        errors = []
        if not Config.ZEP_API_KEY:
            errors.append(t('api.zepApiKeyMissing'))
        if errors:
            logger.error("Configuration errors: %s", errors)
            return jsonify({
                "success": False,
                "error": t('api.configError', details="; ".join(errors))
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
        ProjectManager.save_project(project)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Start background task.
        def build_task():
            set_locale(current_locale)
            build_logger = get_logger('mirofish.build')
            try:
                build_logger.info("[%s] Starting graph build...", task_id)
                task_manager.update_task(
                    task_id, 
                    status=TaskStatus.PROCESSING,
                    message=t('progress.initGraphService')
                )
                
                # Create graph build service.
                builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
                
                # Split text into chunks.
                task_manager.update_task(
                    task_id,
                    message=t('progress.textChunking'),
                    progress=5
                )
                chunks = TextProcessor.split_text(
                    text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                total_chunks = len(chunks)
                
                # Create graph.
                task_manager.update_task(
                    task_id,
                    message=t('progress.creatingZepGraph'),
                    progress=10
                )
                graph_id = run_pipeline_step(
                    "zep_create_graph",
                    lambda gn=graph_name: builder.create_graph(name=gn),
                )

                project.graph_id = graph_id
                ProjectManager.save_project(project)

                task_manager.update_task(
                    task_id,
                    message=t('progress.settingOntology'),
                    progress=15
                )
                run_pipeline_step(
                    "zep_set_ontology",
                    lambda gid=graph_id: builder.set_ontology(gid, ontology),
                )
                
                # Add text. progress_callback signature is (msg, progress_ratio).
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )
                
                task_manager.update_task(
                    task_id,
                    message=t('progress.addingChunks', count=total_chunks),
                    progress=15
                )
                
                episode_uuids = builder.add_text_batches(
                    graph_id,
                    chunks,
                    batch_size=3,
                    progress_callback=add_progress_callback,
                )
                
                # Wait for Zep processing by checking each episode's processed state.
                task_manager.update_task(
                    task_id,
                    message=t('progress.waitingZepProcess'),
                    progress=55
                )
                
                def wait_progress_callback(msg, progress_ratio):
                    progress = 55 + int(progress_ratio * 35)  # 55% - 90%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )
                
                run_pipeline_step(
                    "zep_wait_for_episodes",
                    lambda uuids=episode_uuids: builder._wait_for_episodes(
                        uuids,
                        wait_progress_callback,
                    ),
                )
                
                # Fetch graph data.
                task_manager.update_task(
                    task_id,
                    message=t('progress.fetchingGraphData'),
                    progress=95
                )
                graph_data = run_pipeline_step(
                    "zep_get_graph_data",
                    lambda gid=graph_id: builder.get_graph_data(gid),
                )
                
                # Update project state.
                project.status = ProjectStatus.GRAPH_COMPLETED
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
                task_manager.update_task(
                    task_id,
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
                project.error = str(e)
                ProjectManager.save_project(project)
                
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=t('progress.buildFailed', error=str(e)),
                    error=traceback.format_exc()
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
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": t('api.zepApiKeyMissing')
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
    Delete a Zep graph.
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": t('api.zepApiKeyMissing')
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
