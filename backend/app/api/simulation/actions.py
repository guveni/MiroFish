"""
Simulation-related run and action control API routes
"""

import os
import traceback
import threading
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...config import Config
from ...services.zep_entity_reader import ZepEntityReader
from ...services.oasis_profile_generator import OasisProfileGenerator
from ...services.simulation_manager import (
    SimulationManager,
    SimulationStatus,
    SimulationCancelledError,
)
from ...models.simulation import SimulationRunState
from ...services.simulation_runner import SimulationRunner, RunnerStatus
from ...utils.logger import get_logger
from ...utils.locale import t, get_locale, set_locale
from ...models.project import ProjectManager
from ...models.task import TaskManager, TaskStatus
from ...models.user import User
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.simulation.actions')


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Check if the simulation is already prepared.
    """
    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    
    # Check if the directory exists
    if not os.path.exists(simulation_dir):
        return False, {"reason": "Simulation directory does not exist"}
    
    # List of required files
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv"
    ]
    
    # Check if files exist
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)
    
    if missing_files:
        return False, {
            "reason": "Missing required files",
            "missing_files": missing_files,
            "existing_files": existing_files
        }
    
    # Check status in state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)
        
        logger.debug(f"Detecting simulation preparation status: {simulation_id}, status={status}, config_generated={config_generated}")
        
        prepared_statuses = ["ready", "preparing", "running", "completed", "stopped", "failed"]
        if status in prepared_statuses and config_generated:
            # Get file statistics
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            
            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    profiles_count = len(profiles_data) if isinstance(profiles_data, list) else 0
            
            # If status is "preparing" but files are complete, automatically update status to "ready"
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Automatically updated simulation status: {simulation_id} preparing -> ready")
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Failed to automatically update status: {e}")
            
            logger.info(f"Simulation {simulation_id} detection result: Prepared (status={status}, config_generated={config_generated})")
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files
            }
        else:
            logger.warning(f"Simulation {simulation_id} detection result: Not prepared (status={status}, config_generated={config_generated})")
            return False, {
                "reason": f"Status not in prepared list or config_generated is false: status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated
            }
            
    except Exception as e:
        return False, {"reason": f"Failed to read state file: {str(e)}"}


@router.post('/prepare')
async def prepare_simulation(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Prepare the simulation environment (asynchronous task, LLM intelligently generates all parameters).
    """
    try:
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )
        
        manager = SimulationManager()
        state = await run_in_threadpool(manager.get_simulation, simulation_id)
        
        if not state:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.simulationNotFound', id=simulation_id)
                }
            )
        
        # Check if force regeneration is requested
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"Started processing /prepare request: simulation_id={simulation_id}, force_regenerate={force_regenerate}")
        
        # Check if already prepared (to avoid duplicate generation)
        if not force_regenerate:
            logger.debug(f"Checking if simulation {simulation_id} is already prepared...")
            is_prepared, prepare_info = await run_in_threadpool(_check_simulation_prepared, simulation_id)
            logger.debug(f"Check result: is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"Simulation {simulation_id} is already prepared, skipping duplicate generation")
                return {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                }
            else:
                logger.info(f"Simulation {simulation_id} is not prepared, starting preparation task")
        
        # Get necessary info from project
        project = await run_in_threadpool(ProjectManager.get_project, state.project_id)
        if not project:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.projectNotFound', id=state.project_id)
                }
            )
        
        # Get simulation requirements
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.projectMissingRequirement')
                }
            )
        
        # Get document text
        document_text = await run_in_threadpool(ProjectManager.get_extracted_text, state.project_id) or ""
        
        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = int(data.get('parallel_profile_count', Config.SIM_PROFILE_PARALLEL_COUNT))
        
        # ========== Synchronously retrieve entity count (before background task starts) ==========
        try:
            logger.info(f"Synchronously retrieving entity count: graph_id={state.graph_id}")
            reader = ZepEntityReader()
            filtered_preview = await run_in_threadpool(
                reader.filter_defined_entities,
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False
            )
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"Expected entities count: {filtered_preview.filtered_count}, types: {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"Failed to synchronously retrieve entity count (will retry in background task): {e}")
        
        # Create asynchronous task
        task_manager = TaskManager()
        task_id = await run_in_threadpool(
            task_manager.create_task,
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )
        
        # Update simulation status
        state.status = SimulationStatus.PREPARING
        await run_in_threadpool(manager._save_simulation_state, state)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Define background task
        def run_prepare():
            set_locale(current_locale)
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message=t('progress.startPreparingEnv')
                )
                
                stage_details = {}
                
                def progress_callback(stage, progress, message, **kwargs):
                    stage_weights = {
                        "reading": (0, 20),
                        "generating_profiles": (20, 70),
                        "generating_config": (70, 90),
                        "copying_scripts": (90, 100)
                    }
                    
                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)
                    
                    stage_names = {
                        "reading": t('progress.readingGraphEntities'),
                        "generating_profiles": t('progress.generatingProfiles'),
                        "generating_config": t('progress.generatingSimConfig'),
                        "copying_scripts": t('progress.preparingScripts')
                    }
                    
                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)
                    
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }
                    
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }
                    
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"
                    
                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )
                
                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count
                )
                
                task_manager.complete_task(
                    task_id,
                    result=result_state.to_simple_dict()
                )
                
            except SimulationCancelledError:
                logger.info(
                    "Preparation cancelled for simulation %s (deleted or stopped)",
                    simulation_id,
                )
                task_manager.fail_task(task_id, "Simulation deleted")
            except Exception as e:
                if manager.is_cancelled(simulation_id):
                    logger.info(
                        "Preparation aborted for deleted simulation %s",
                        simulation_id,
                    )
                    task_manager.fail_task(task_id, "Simulation deleted")
                    return
                logger.error(f"Failed to prepare simulation: {str(e)}")
                task_manager.fail_task(task_id, str(e))
                
                # Update simulation status to failed
                state_to_fail = manager.get_simulation(simulation_id)
                if state_to_fail:
                    state_to_fail.status = SimulationStatus.FAILED
                    state_to_fail.error = str(e)
                    manager._save_simulation_state(state_to_fail)
        
        # Start background thread
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()
        
        return {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": t('api.prepareStarted'),
                "already_prepared": False,
                "expected_entities_count": state.entities_count,
                "entity_types": state.entity_types
            }
        }
        
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start preparation task: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/prepare/status')
async def get_prepare_status(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Query preparation task progress.
    """
    try:
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # If simulation_id is provided, check if preparation is complete first
        if simulation_id:
            is_prepared, prepare_info = await run_in_threadpool(_check_simulation_prepared, simulation_id)
            if is_prepared:
                return {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                }
        
        # If no task_id, return status of not started
        if not task_id:
            if simulation_id:
                return {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": t('api.notStartedPrepare'),
                        "already_prepared": False
                    }
                }
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireTaskOrSimId')
                }
            )
        
        task_manager = TaskManager()
        task = await run_in_threadpool(task_manager.get_task, task_id)
        
        if not task:
            if simulation_id:
                is_prepared, prepare_info = await run_in_threadpool(_check_simulation_prepared, simulation_id)
                if is_prepared:
                    return {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": t('api.taskCompletedPrepared'),
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    }
            
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.taskNotFound', id=task_id)
                }
            )
        
        task_dict = task.to_dict()
        task_dict["already_prepared"] = False
        
        return {
            "success": True,
            "data": task_dict
        }
        
    except Exception as e:
        logger.error(f"Failed to query task status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.post('/start')
async def start_simulation(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Start running the simulation.
    """
    try:
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        platform = data.get('platform', 'parallel')
        max_rounds = data.get('max_rounds')
        enable_graph_memory_update = data.get('enable_graph_memory_update', False)
        force = data.get('force', False)

        # Validate max_rounds parameter
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "error": t('api.maxRoundsPositive')
                        }
                    )
            except (ValueError, TypeError):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.maxRoundsInvalid')
                    }
                )

        if platform not in ['twitter', 'reddit', 'parallel']:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.invalidPlatform', platform=platform)
                }
            )

        # Check if the simulation is ready
        manager = SimulationManager()
        state = await run_in_threadpool(manager.get_simulation, simulation_id)

        if not state:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.simulationNotFound', id=simulation_id)
                }
            )

        force_restarted = False
        
        if state.status != SimulationStatus.READY:
            is_prepared, prepare_info = await run_in_threadpool(_check_simulation_prepared, simulation_id)

            if is_prepared:
                if state.status == SimulationStatus.RUNNING:
                    run_state = await run_in_threadpool(SimulationRunner.get_run_state, simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        if force:
                            logger.info(f"Force mode: stopping running simulation {simulation_id}")
                            try:
                                await run_in_threadpool(SimulationRunner.stop_simulation, simulation_id)
                            except Exception as e:
                                logger.warning(f"Warning occurred when stopping simulation: {str(e)}")
                        else:
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "success": False,
                                    "error": t('api.simRunningForceHint')
                                }
                            )

                if force:
                    logger.info(f"Force mode: cleaning up simulation logs {simulation_id}")
                    cleanup_result = await run_in_threadpool(SimulationRunner.cleanup_simulation_logs, simulation_id)
                    if not cleanup_result.get("success"):
                        logger.warning(f"Warning occurred when cleaning up logs: {cleanup_result.get('errors')}")
                    force_restarted = True

                logger.info(f"Simulation {simulation_id} preparation completed, resetting status to ready (original status: {state.status.value})")
                state.status = SimulationStatus.READY
                await run_in_threadpool(manager._save_simulation_state, state)
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.simNotReady', status=state.status.value)
                    }
                )
        
        # Get graph ID
        graph_id = None
        if enable_graph_memory_update:
            graph_id = state.graph_id
            if not graph_id:
                project = await run_in_threadpool(ProjectManager.get_project, state.project_id)
                if project:
                    graph_id = project.graph_id
            
            if not graph_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.graphIdRequiredForMemory')
                    }
                )
            
            logger.info(f"Enabled graph memory update: simulation_id={simulation_id}, graph_id={graph_id}")
        
        # Start simulation
        if Config.SIMULATION_DISTRIBUTED_WORKERS:
            logger.info(f"Dispatching simulation {simulation_id} to distributed Inngest workers...")
            
            run_state = await run_in_threadpool(SimulationRunner.get_run_state, simulation_id)
            if not run_state:
                run_state = SimulationRunState(
                    simulation_id=simulation_id,
                    runner_status=RunnerStatus.STARTING,
                )
            else:
                run_state.runner_status = RunnerStatus.STARTING
                run_state.error = None
            
            run_state.started_at = datetime.now().isoformat()
            
            if platform == "twitter":
                run_state.twitter_running = True
            elif platform == "reddit":
                run_state.reddit_running = True
            else:
                run_state.twitter_running = True
                run_state.reddit_running = True
                
            await run_in_threadpool(SimulationRunner._save_run_state, run_state)

            from ...inngest_client import inngest_client
            import inngest
            inngest_client.send_sync(
                inngest.Event(
                    name="simulation/run",
                    data={
                        "simulation_id": simulation_id,
                        "platform": platform,
                        "max_rounds": max_rounds,
                        "enable_graph_memory_update": enable_graph_memory_update,
                        "graph_id": graph_id,
                        "locale": get_locale(),
                    }
                )
            )
        else:
            run_state = await run_in_threadpool(
                SimulationRunner.start_simulation,
                simulation_id=simulation_id,
                platform=platform,
                max_rounds=max_rounds,
                enable_graph_memory_update=enable_graph_memory_update,
                graph_id=graph_id
            )
        
        state.status = SimulationStatus.RUNNING
        await run_in_threadpool(manager._save_simulation_state, state)
        
        response_data = run_state.to_dict()
        if max_rounds:
            response_data['max_rounds_applied'] = max_rounds
        response_data['graph_memory_update_enabled'] = enable_graph_memory_update
        response_data['force_restarted'] = force_restarted
        if enable_graph_memory_update:
            response_data['graph_id'] = graph_id
        
        return {
            "success": True,
            "data": response_data
        }
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/stop')
async def stop_simulation(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Stop the simulation.
    """
    try:
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )
        
        run_state = await run_in_threadpool(SimulationRunner.stop_simulation, simulation_id)
        
        # Update simulation status
        manager = SimulationManager()
        state = await run_in_threadpool(manager.get_simulation, simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            await run_in_threadpool(manager._save_simulation_state, state)
        
        return {
            "success": True,
            "data": run_state.to_dict()
        }
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop simulation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/run-status')
async def get_run_status(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time running status of the simulation (used for frontend polling).
    """
    try:
        run_state = await run_in_threadpool(SimulationRunner.get_run_state, simulation_id)
        
        if not run_state:
            return {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "current_round": 0,
                    "total_rounds": 0,
                    "progress_percent": 0,
                    "twitter_actions_count": 0,
                    "reddit_actions_count": 0,
                    "total_actions_count": 0,
                }
            }
        
        return {
            "success": True,
            "data": run_state.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get running status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/run-status/detail')
async def get_run_status_detail(
    simulation_id: str,
    platform: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed running status of the simulation (containing all actions).
    """
    try:
        run_state = await run_in_threadpool(SimulationRunner.get_run_state, simulation_id)
        
        if not run_state:
            return {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "all_actions": [],
                    "twitter_actions": [],
                    "reddit_actions": []
                }
            }
        
        # Get complete list of actions
        all_actions = await run_in_threadpool(
            SimulationRunner.get_all_actions,
            simulation_id=simulation_id,
            platform=platform
        )
        
        # Get actions by platform
        twitter_actions = await run_in_threadpool(
            SimulationRunner.get_all_actions,
            simulation_id=simulation_id,
            platform="twitter"
        ) if not platform or platform == "twitter" else []
        
        reddit_actions = await run_in_threadpool(
            SimulationRunner.get_all_actions,
            simulation_id=simulation_id,
            platform="reddit"
        ) if not platform or platform == "reddit" else []
        
        # Get actions of the current round
        current_round = run_state.current_round
        recent_actions = await run_in_threadpool(
            SimulationRunner.get_all_actions,
            simulation_id=simulation_id,
            platform=platform,
            round_num=current_round
        ) if current_round > 0 else []
        
        # Get basic status information
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        result["recent_actions"] = [a.to_dict() for a in recent_actions]
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to get detailed status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/actions')
async def get_simulation_actions(
    simulation_id: str,
    limit: int = Query(100),
    offset: int = Query(0),
    platform: Optional[str] = Query(None),
    agent_id: Optional[int] = Query(None),
    round_num: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get Agent action history in the simulation.
    """
    try:
        actions = await run_in_threadpool(
            SimulationRunner.get_actions,
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        return {
            "success": True,
            "data": {
                "count": len(actions),
                "actions": [a.to_dict() for a in actions]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get action history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/timeline')
async def get_simulation_timeline(
    simulation_id: str,
    start_round: int = Query(0),
    end_round: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get simulation timeline (summarized by rounds).
    """
    try:
        timeline = await run_in_threadpool(
            SimulationRunner.get_timeline,
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round
        )
        
        return {
            "success": True,
            "data": {
                "rounds_count": len(timeline),
                "timeline": timeline
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get timeline: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/generate-profiles')
async def generate_profiles(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Directly generate OASIS Agent Profile from graph (without creating simulation).
    """
    try:
        graph_id = data.get('graph_id')
        if not graph_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireGraphId')
                }
            )
        
        entity_types = data.get('entity_types')
        use_llm = data.get('use_llm', True)
        platform = data.get('platform', 'reddit')
        
        reader = ZepEntityReader()
        filtered = await run_in_threadpool(
            reader.filter_defined_entities,
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=True
        )
        
        if filtered.filtered_count == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.noMatchingEntities')
                }
            )
        
        generator = OasisProfileGenerator()
        profiles = await run_in_threadpool(
            generator.generate_profiles_from_entities,
            entities=filtered.entities,
            use_llm=use_llm
        )
        
        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]
        
        return {
            "success": True,
            "data": {
                "platform": platform,
                "entity_types": list(filtered.entity_types),
                "count": len(profiles_data),
                "profiles": profiles_data
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate Profile: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/env-status')
async def get_env_status(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Get simulation environment status.
    """
    try:
        simulation_id = data.get('simulation_id')
        
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        env_alive = await run_in_threadpool(SimulationRunner.check_env_alive, simulation_id)
        env_status = await run_in_threadpool(SimulationRunner.get_env_status_detail, simulation_id)

        if env_alive:
            message = t('api.envRunning')
        else:
            message = t('api.envNotRunningShort')

        return {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "env_alive": env_alive,
                "twitter_available": env_status.get("twitter_available", False),
                "reddit_available": env_status.get("reddit_available", False),
                "message": message
            }
        }

    except Exception as e:
        logger.error(f"Failed to get environment status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/close-env')
async def close_simulation_env(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Close simulation environment.
    """
    try:
        simulation_id = data.get('simulation_id')
        timeout = data.get('timeout', 30)
        
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )
        
        result = await run_in_threadpool(
            SimulationRunner.close_simulation_env,
            simulation_id=simulation_id,
            timeout=timeout
        )
        
        # Update simulation status
        manager = SimulationManager()
        state = await run_in_threadpool(manager.get_simulation, simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            await run_in_threadpool(manager._save_simulation_state, state)
        
        return {
            "success": result.get("success", False),
            "data": result
        }
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to close environment: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
