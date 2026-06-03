"""
Simulation-related CRUD API routes
"""

import os
import traceback
import json
import csv
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from ...config import Config
from ...services import _backend
from ...services.zep_entity_reader import ZepEntityReader
from ...services.simulation_manager import SimulationManager, SimulationStatus
from ...services.simulation_runner import SimulationRunner, RunnerStatus
from ...utils.logger import get_logger
from ...utils.locale import t
from ...models.project import ProjectManager
from ...models.user import User
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.simulation.crud')


def _graph_backend_error_response():
    ok, error_key = _backend.is_available()
    if ok:
        return None
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": t(error_key or 'api.graphBackendUnavailable')
        }
    )


def _get_report_id_for_simulation(simulation_id: str) -> Optional[str]:
    """
    Get the latest report_id associated with the simulation.
    
    Traverses the reports directory, finds reports matching the simulation_id,
    and returns the latest one if multiple exist (sorted by created_at).
    """
    reports_dir = os.path.join(os.path.dirname(__file__), '../../../uploads/reports')
    if not os.path.exists(reports_dir):
        return None
    
    matching_reports = []
    
    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue
            
            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append({
                        "report_id": meta.get("report_id"),
                        "created_at": meta.get("created_at", ""),
                        "status": meta.get("status", "")
                    })
            except Exception:
                continue
        
        if not matching_reports:
            return None
        
        # Sort in descending order of creation time and return the latest
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")
        
    except Exception as e:
        logger.warning(f"Failed to find report for simulation {simulation_id}: {e}")
        return None


# ============== Simulation CRUD Interface ==============

@router.post('/create')
async def create_simulation(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new simulation.
    """
    try:
        project_id = data.get('project_id')
        if not project_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireProjectId')
                }
            )
        
        project = await run_in_threadpool(ProjectManager.get_project, project_id)
        if not project:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.projectNotFound', id=project_id)
                }
            )
        
        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.graphNotBuilt')
                }
            )
        
        manager = SimulationManager()
        state = await run_in_threadpool(
            manager.create_simulation,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )
        
        return {
            "success": True,
            "data": state.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to create simulation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.delete('/{simulation_id}')
async def delete_simulation(
    simulation_id: str,
    delete_project: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a simulation and its associated data.
    """
    try:
        # Handle deletion of project-only history cards (with synthetic simulation_id = sim_pending_<project_id>)
        if simulation_id.startswith("sim_pending_"):
            project_id = simulation_id.replace("sim_pending_", "")
            project = await run_in_threadpool(ProjectManager.get_project, project_id)
            project_deleted = False
            graph_deleted = False
            if project:
                if project.graph_id:
                    try:
                        from ..graph import _get_graph_backend
                        backend = _get_graph_backend()
                        await run_in_threadpool(backend.delete_graph, project.graph_id)
                        graph_deleted = True
                    except Exception as e:
                        logger.warning(f"Failed to delete graph {project.graph_id}: {e}")
                
                deleted = await run_in_threadpool(ProjectManager.delete_project, project_id)
                if deleted:
                    project_deleted = True
            return {
                "success": True,
                "data": {
                    "simulation_deleted": False,
                    "project_deleted": project_deleted,
                    "graph_deleted": graph_deleted
                }
            }

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

        # Stop in-flight preparation and simulation runs before removing data.
        await run_in_threadpool(SimulationManager.request_cancel, simulation_id)

        from ...models.task import TaskManager, TaskStatus
        task_manager = TaskManager()
        tasks = await run_in_threadpool(task_manager.list_tasks, "simulation_prepare")
        for task in tasks:
            if task.get("metadata", {}).get("simulation_id") != simulation_id:
                continue
            if task.get("status") in (
                TaskStatus.PENDING.value,
                TaskStatus.PROCESSING.value,
            ):
                await run_in_threadpool(task_manager.fail_task, task["task_id"], "Simulation deleted")

        try:
            run_state = await run_in_threadpool(SimulationRunner.get_run_state, simulation_id)
            if run_state and run_state.runner_status in (
                RunnerStatus.RUNNING,
                RunnerStatus.PAUSED,
                RunnerStatus.STARTING,
            ):
                await run_in_threadpool(SimulationRunner.stop_simulation, simulation_id)
        except ValueError:
            pass
        except Exception as e:
            logger.warning(
                "Failed to stop simulation runner for %s: %s",
                simulation_id,
                e,
            )
        
        # Delete simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        if os.path.exists(sim_dir):
            try:
                import shutil
                await run_in_threadpool(shutil.rmtree, sim_dir)
            except Exception as e:
                logger.warning(f"Failed to delete simulation directory {sim_dir}: {e}")

        # Remove from in-memory cache
        await run_in_threadpool(manager._simulations.pop, simulation_id, None)
                
        # Delete run checkpoints if any
        try:
            from ...services.run_checkpoint_store import delete_simulation_checkpoints
            await run_in_threadpool(delete_simulation_checkpoints, simulation_id)
        except Exception as e:
            logger.warning(f"Failed to delete simulation checkpoints: {e}")
                
        # If requested, also delete the project and graph
        project_deleted = False
        graph_deleted = False
        if delete_project and state.project_id:
            try:
                project = await run_in_threadpool(ProjectManager.get_project, state.project_id)
                
                if project:
                    # Try to delete graph first
                    if project.graph_id:
                        try:
                            from ..graph import _get_graph_backend
                            backend = _get_graph_backend()
                            await run_in_threadpool(backend.delete_graph, project.graph_id)
                            graph_deleted = True
                        except Exception as e:
                            logger.warning(f"Failed to delete graph {project.graph_id}: {e}")
                            
                    # Then delete project
                    deleted = await run_in_threadpool(ProjectManager.delete_project, state.project_id)
                    if deleted:
                        project_deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete project {state.project_id}: {e}")
                
        return {
            "success": True,
            "data": {
                "simulation_deleted": True,
                "project_deleted": project_deleted,
                "graph_deleted": graph_deleted
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to delete simulation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get('/list')
async def list_simulations(
    project_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    List all simulations.
    """
    try:
        if project_id:
            project = await run_in_threadpool(ProjectManager.get_project, project_id)
            if not project or (project.user_id and project.user_id != current_user.id):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "error": "Access denied to this project."
                    }
                )
            manager = SimulationManager()
            simulations = await run_in_threadpool(manager.list_simulations, project_id=project_id)
        else:
            # Filter simulations to only those belonging to the current user's projects
            user_projects = await run_in_threadpool(ProjectManager.list_projects, user_id=current_user.id, limit=500)
            user_project_ids = {p.project_id for p in user_projects}
            
            manager = SimulationManager()
            all_simulations = await run_in_threadpool(manager.list_simulations)
            simulations = [s for s in all_simulations if s.project_id in user_project_ids]
        
        return {
            "success": True,
            "data": [s.to_dict() for s in simulations],
            "count": len(simulations)
        }
        
    except Exception as e:
        logger.error(f"Failed to list simulations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/history')
async def get_simulation_history(
    limit: int = Query(20),
    current_user: User = Depends(get_current_user)
):
    """
    Get the simulation history list (with project details).
    """
    try:
        manager = SimulationManager()
        all_simulations = await run_in_threadpool(manager.list_simulations)
        
        # Load user projects to see what we already own
        projects = await run_in_threadpool(ProjectManager.list_projects, user_id=current_user.id, limit=max(100, limit * 2))
        user_project_ids = {p.project_id for p in projects}
        
        # Auto-recover/associate any orphaned simulations/projects
        from ...models.project import Project, ProjectStatus
        from ...database import db
        
        db_updated = False
        for sim in all_simulations:
            if not sim.project_id:
                continue
                
            if sim.project_id not in user_project_ids:
                # Check if the project already exists in the DB (but belongs to no one or someone else)
                existing_proj = await run_in_threadpool(ProjectManager.get_project, sim.project_id)
                if existing_proj:
                    if not existing_proj.user_id:
                        existing_proj.user_id = current_user.id
                        await run_in_threadpool(ProjectManager.save_project, existing_proj)
                        user_project_ids.add(sim.project_id)
                else:
                    # Create a recovered project for this simulation
                    try:
                        config = await run_in_threadpool(manager.get_simulation_config, sim.simulation_id)
                        req = config.get("simulation_requirement", "") if config else ""
                        
                        proj_name = ""
                        if req:
                            clean_req = " ".join(req.split())
                            proj_name = clean_req[:40] + ("..." if len(clean_req) > 40 else "")
                        if not proj_name:
                            proj_name = f"Recovered Project ({sim.simulation_id[:8]})"
                            
                        now = datetime.now().isoformat()
                        created_at = getattr(sim, "created_at", now) or now
                        
                        project = Project(
                            project_id=sim.project_id,
                            user_id=current_user.id,
                            name=proj_name,
                            status=ProjectStatus.GRAPH_COMPLETED.value,
                            created_at=created_at,
                            updated_at=now,
                            simulation_requirement=req,
                            graph_id=config.get("graph_id") if config else None,
                            files=[]
                        )
                        db.session.add(project)
                        db_updated = True
                        user_project_ids.add(sim.project_id)
                    except Exception as e:
                        logger.error(f"Failed to auto-recover project {sim.project_id}: {e}")
                        
        if db_updated:
            await run_in_threadpool(db.session.commit)
            # Reload projects after recovery
            projects = await run_in_threadpool(ProjectManager.list_projects, user_id=current_user.id, limit=max(100, limit * 2))
            user_project_ids = {p.project_id for p in projects}
            
        # Filter simulations to only those belonging to the current user's projects
        user_simulations = [sim for sim in all_simulations if sim.project_id in user_project_ids]
        
        simulated_project_ids = {sim.project_id for sim in user_simulations if sim.project_id}
        
        enriched_items = []
        for sim in user_simulations:
            sim_dict = sim.to_dict()
            
            # Get simulation configuration info (reading simulation_requirement from simulation_config.json)
            config = await run_in_threadpool(manager.get_simulation_config, sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get("simulation_requirement", "")
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get("total_simulation_hours", 0)
                # Recommended rounds (fallback value)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0) * 60 / 
                    max(time_config.get("minutes_per_round", 30), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0
            
            # Get running status (reading actual rounds configured by user from run_state.json)
            run_state = await run_in_threadpool(SimulationRunner.get_run_state, sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Use total_rounds set by user; if none, use recommended rounds
                sim_dict["total_rounds"] = run_state.total_rounds if run_state.total_rounds > 0 else recommended_rounds
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds
            
            # Get the file list of associated projects (maximum 3 files)
            project = await run_in_threadpool(ProjectManager.get_project, sim.project_id)
            if project:
                sim_dict["project_name"] = project.name
                sim_dict["files"] = await run_in_threadpool(
                    ProjectManager.get_display_files, project, 3
                )
            else:
                sim_dict["project_name"] = "Unknown Project"
                sim_dict["files"] = []
            
            # Get the associated report_id (find the latest report for this simulation)
            sim_dict["report_id"] = await run_in_threadpool(_get_report_id_for_simulation, sim.simulation_id)
            
            # Add version number
            sim_dict["version"] = "v1.0.2"
            
            # Format date
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""
            
            enriched_items.append(sim_dict)
            
        # Also add projects that do not have an associated simulation yet
        for proj in projects:
            if proj.project_id not in simulated_project_ids:
                proj_dict = {
                    "simulation_id": f"sim_pending_{proj.project_id}",
                    "project_id": proj.project_id,
                    "project_name": proj.name,
                    "simulation_requirement": proj.simulation_requirement or "",
                    "status": proj.status,
                    "entities_count": 0,
                    "profiles_count": 0,
                    "entity_types": [],
                    "created_at": proj.created_at,
                    "updated_at": proj.updated_at,
                    "total_rounds": 0,
                    "current_round": 0,
                    "runner_status": "idle",
                    "report_id": None,
                    "version": "v1.0.2",
                    "files": await run_in_threadpool(
                        ProjectManager.get_display_files, proj, 3
                    )
                }
                try:
                    proj_dict["created_date"] = proj.created_at[:10]
                except:
                    proj_dict["created_date"] = ""
                
                enriched_items.append(proj_dict)
                
        # Sort items descending by created_at safely
        enriched_items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        result_items = enriched_items[:limit]
        
        return {
            "success": True,
            "data": result_items,
            "count": len(result_items)
        }
        
    except Exception as e:
        logger.error(f"Failed to get simulation history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}')
async def get_simulation(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get simulation status."""
    try:
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
        
        result = state.to_dict()
        
        # If simulation is ready, attach running instructions
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = await run_in_threadpool(manager.get_run_instructions, simulation_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to get simulation status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/profiles')
async def get_simulation_profiles(
    simulation_id: str,
    platform: str = Query('reddit'),
    current_user: User = Depends(get_current_user)
):
    """
    Get Agent Profiles of the simulation.
    """
    try:
        manager = SimulationManager()
        profiles = await run_in_threadpool(manager.get_profiles, simulation_id, platform=platform)
        
        return {
            "success": True,
            "data": {
                "platform": platform,
                "count": len(profiles),
                "profiles": profiles
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
        logger.error(f"Failed to get Profile: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/config')
async def get_simulation_config(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get simulation configuration.
    """
    try:
        manager = SimulationManager()
        config = await run_in_threadpool(manager.get_simulation_config, simulation_id)
        
        if not config:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.configNotFound')
                }
            )
        
        return {
            "success": True,
            "data": config
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/config/download')
async def download_simulation_config(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download simulation configuration file."""
    try:
        manager = SimulationManager()
        sim_dir = await run_in_threadpool(manager._get_simulation_dir, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.configFileNotFound')
                }
            )
        
        return FileResponse(
            config_path,
            filename="simulation_config.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to download configuration: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/script/{script_name}/download')
async def download_simulation_script(
    script_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download simulation run script file (generic scripts, located in backend/scripts/).
    """
    try:
        # Scripts are located in backend/scripts/ directory
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts'))
        
        # Validate script name
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py", 
            "run_parallel_simulation.py",
            "action_logger.py"
        ]
        
        if script_name not in allowed_scripts:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.unknownScript', name=script_name, allowed=allowed_scripts)
                }
            )
        
        script_path = os.path.join(scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.scriptFileNotFound', name=script_name)
                }
            )
        
        return FileResponse(
            script_path,
            filename=script_name,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Failed to download script: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
