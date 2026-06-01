"""
Durable background execution tasks using Inngest.
"""

import os
import time
import traceback
from datetime import datetime
import inngest

from ..inngest_client import inngest_client
from ..database import db
from ..config import Config
from ..models.project import ProjectManager, ProjectStatus
from ..models.task import TaskManager, TaskStatus
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..services import _backend
from ..services.research_query_generator import ResearchQueryGenerator
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..utils.locale import t, set_locale
from ..services.run_checkpoint_store import checkpoint_project_stage
from ..services.web_search import (
    EMPTY_GEMINI_WEB_SEARCH_RESULTS,
    GEMINI_WEB_SEARCH_MODEL_NOT_SET,
    GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED,
    search_queries_to_corpus,
)


@inngest_client.create_function(
    fn_id="generate_ontology_task",
    trigger=inngest.TriggerEvent(event="ontology/generate"),
)
def generate_ontology_task(ctx: inngest.ContextSync) -> dict:
    """Inngest background function to generate ontology."""
    data = ctx.event.data
    project_id = data["project_id"]
    task_id = data["task_id"]
    simulation_requirement = data["simulation_requirement"]
    additional_context = data.get("additional_context", "")
    use_vertex_search = data.get("use_vertex_search", False)
    locale_str = data.get("locale", "en")

    set_locale(locale_str)
    worker_logger = get_logger('mirofish.ontology')
    task_manager = TaskManager()

    # We must load the local project inside the session
    local_project = ProjectManager.get_project(project_id)
    if not local_project:
        task_manager.fail_task(task_id, f"Project not found: {project_id}")
        return {"success": False, "error": f"Project not found: {project_id}"}

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
                
                def search_progress(completed: int, total: int):
                    task_manager.update_task(
                        task_id,
                        progress=30 + int(completed / max(1, total) * 10),
                    )
                    
                search_corpus, search_metadata = search_queries_to_corpus(
                    queries,
                    simulation_requirement=simulation_requirement,
                    progress_callback=search_progress,
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

        # Prompt-only runs: use simulation_requirement as seed text when no files were uploaded.
        if not document_texts and simulation_requirement and simulation_requirement.strip():
            req_text = simulation_requirement.strip()
            document_texts.append(req_text)
            all_text += f"\n\n=== simulation_requirement ===\n{req_text}"

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
        local_project.status = ProjectStatus.ONTOLOGY_GENERATED.value
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

        # Build payload
        result_payload = {
            "project_id": local_project.project_id,
            "project_name": local_project.name,
            "ontology": local_project.ontology,
            "analysis_summary": local_project.analysis_summary,
            "files": local_project.files,
            "total_text_length": local_project.total_text_length,
            "gemini_grounding_metadata": local_project.gemini_grounding_metadata,
        }
        if use_vertex_search and search_metadata is not None:
            result_payload["search_metadata"] = search_metadata

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
        return {"success": True, "project_id": project_id}

    except Exception as e:
        worker_logger.exception("[%s] Ontology generation failed", task_id)
        err_msg = str(e)
        local_project.status = ProjectStatus.FAILED.value
        local_project.error = err_msg
        local_project.ontology_task_id = None
        ProjectManager.save_project(local_project)
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=t('progress.ontologyFailed', error=err_msg),
            error=traceback.format_exc(),
        )
        return {"success": False, "error": err_msg}


@inngest_client.create_function(
    fn_id="build_graph_task",
    trigger=inngest.TriggerEvent(event="graph/build"),
)
def build_graph_task(ctx: inngest.ContextSync) -> dict:
    """Inngest background function to build graph."""
    data = ctx.event.data
    project_id = data["project_id"]
    task_id = data["task_id"]
    graph_name = data["graph_name"]
    chunk_size = data["chunk_size"]
    chunk_overlap = data["chunk_overlap"]
    locale_str = data.get("locale", "en")

    set_locale(locale_str)
    build_logger = get_logger('mirofish.build')
    task_manager = TaskManager()

    project = ProjectManager.get_project(project_id)
    if not project:
        task_manager.fail_task(task_id, f"Project not found: {project_id}")
        return {"success": False, "error": f"Project not found: {project_id}"}

    def _persist_build_progress(progress: int, message: str) -> None:
        project.graph_build_progress = max(int(project.graph_build_progress or 0), int(progress))
        if message:
            project.graph_build_message = message
        ProjectManager.save_project(project)

    def update_build_task(*, progress=None, message=None, **kwargs):
        task_manager.update_task(
            task_id,
            progress=progress,
            message=message,
            **kwargs,
        )
        if progress is not None or message:
            _persist_build_progress(
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
        
        # Load extracted text.
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            raise ValueError(t('api.textNotFound'))
            
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
        from ..utils.pipeline_retry import run_pipeline_step
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
            lambda gid=graph_id: builder.set_ontology(gid, project.ontology),
        )
        
        # Add text
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
        
        # Wait for processing
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
        
        # Fetch graph data
        update_build_task(
            message=t('progress.fetchingGraphData'),
            progress=95,
        )
        graph_data = run_pipeline_step(
            "zep_get_graph_data",
            lambda gid=graph_id: builder.get_graph_data(gid),
        )
        
        # Update project state
        project.status = ProjectStatus.GRAPH_COMPLETED.value
        project.graph_build_task_id = None
        _persist_build_progress(
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
        
        # Complete task
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

        return {"success": True, "graph_id": graph_id}
        
    except Exception as e:
        build_logger.error("[%s] Graph build failed: %s", task_id, str(e))
        build_logger.debug(traceback.format_exc())
        
        project.status = ProjectStatus.FAILED.value
        project.graph_build_task_id = None
        project.error = str(e)
        _persist_build_progress(
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

        return {"success": False, "error": str(e)}


@inngest_client.create_function(
    fn_id="generate_report_task",
    trigger=inngest.TriggerEvent(event="report/generate"),
)
def generate_report_task(ctx: inngest.ContextSync) -> dict:
    """Inngest background function to generate report."""
    data = ctx.event.data
    simulation_id = data["simulation_id"]
    graph_id = data["graph_id"]
    report_id = data["report_id"]
    simulation_requirement = data["simulation_requirement"]
    task_id = data["task_id"]
    locale_str = data.get("locale", "en")

    set_locale(locale_str)
    task_manager = TaskManager()

    try:
        task_manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            progress=0,
            message=t('api.initReportAgent')
        )
        
        # Create Report Agent
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement
        )
        
        # Progress callback
        def progress_callback(stage, progress, message):
            task_manager.update_task(
                task_id,
                progress=progress,
                message=f"[{stage}] {message}"
            )
        
        # Generate report
        report = agent.generate_report(
            progress_callback=progress_callback,
            report_id=report_id
        )
        
        # Save report
        ReportManager.save_report(report)
        
        if report.status == ReportStatus.COMPLETED:
            task_manager.complete_task(
                task_id,
                result={
                    "report_id": report.report_id,
                    "simulation_id": simulation_id,
                    "status": "completed"
                }
            )
            return {"success": True, "report_id": report_id}
        else:
            task_manager.fail_task(task_id, report.error or t('api.reportGenerateFailed'))
            return {"success": False, "error": report.error or "Generation failed"}
        
    except Exception as e:
        get_logger('mirofish.api.report').error(f"Inngest report generation failed: {str(e)}")
        task_manager.fail_task(task_id, str(e))
        
        try:
            failed_report = ReportManager.get_report(report_id)
            if failed_report and failed_report.status != ReportStatus.COMPLETED:
                failed_report.status = ReportStatus.FAILED
                failed_report.error = str(e)
                ReportManager.save_report(failed_report)
                ReportManager.update_progress(report_id, "failed", -1, str(e), completed_sections=[])
        except Exception as save_ex:
            get_logger('mirofish.api.report').error(f"Failed to update report status to FAILED in Inngest task: {save_ex}")
        
        return {"success": False, "error": str(e)}


@inngest_client.create_function(
    fn_id="run_simulation_task",
    trigger=inngest.TriggerEvent(event="simulation/run"),
)
def run_simulation_task(ctx: inngest.ContextSync) -> dict:
    """Inngest background function to run social simulations as a distributed task."""
    data = ctx.event.data
    simulation_id = data["simulation_id"]
    platform = data["platform"]
    max_rounds = data.get("max_rounds")
    enable_graph_memory_update = data.get("enable_graph_memory_update", False)
    graph_id = data.get("graph_id")
    locale_str = data.get("locale", "en")

    set_locale(locale_str)
    worker_logger = get_logger('mirofish.simulation_worker')
    worker_logger.info(f"=== Starting Distributed Simulation Task (Inngest) === simulation_id={simulation_id}")

    try:
        from ..services.simulation_runner import SimulationRunner
        
        # Start simulation execution on the worker
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id,
            allow_starting=True
        )
        
        # Monitor the execution synchronously inside Inngest to keep the task active
        sim_dir = os.path.join(SimulationRunner.RUN_STATE_DIR, simulation_id)
        process = SimulationRunner._processes.get(simulation_id)
        
        if process:
            worker_logger.info(f"Inngest simulation process started on worker. PID: {process.pid}")
            # Keep task alive and wait for process to finish
            while process.poll() is None:
                time.sleep(5)
                
            exit_code = process.returncode
            worker_logger.info(f"Distributed simulation process completed with exit code: {exit_code}")
            return {"success": True, "simulation_id": simulation_id, "exit_code": exit_code}
        else:
            return {"success": False, "error": "Process failed to start on worker"}
            
    except Exception as e:
        worker_logger.error(f"Inngest distributed simulation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

