"""
Real-time API routes for simulation profiles and configurations.
"""

import os
import json
import csv
import traceback
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...config import Config
from ...utils.logger import get_logger
from ...utils.locale import t
from ...models.user import User
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.simulation.realtime')


@router.get('/{simulation_id}/profiles/realtime')
async def get_simulation_profiles_realtime(
    simulation_id: str,
    platform: str = Query('reddit'),
    current_user: User = Depends(get_current_user)
):
    """
    Real-time retrieval of simulation Agent Profiles (used to view progress in real-time during generation).
    """
    try:
        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.simulationNotFound', id=simulation_id)
                }
            )
        
        # Determine file path
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")
        
        # Check if file exists
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None
        
        if file_exists:
            # Get file modification time
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                if platform == "reddit":
                    def read_json():
                        with open(profiles_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    profiles = await run_in_threadpool(read_json)
                else:
                    def read_csv():
                        with open(profiles_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            return list(reader)
                    profiles = await run_in_threadpool(read_csv)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read profiles file (might be in write mode): {e}")
                profiles = []
        
        # Check if generation is in progress (judged via state.json)
        is_generating = False
        total_expected = None
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                def read_state():
                    with open(state_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                state_data = await run_in_threadpool(read_state)
                status_val = state_data.get("status", "")
                is_generating = status_val == "preparing"
                total_expected = state_data.get("entities_count")
            except Exception:
                pass
        
        return {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "platform": platform,
                "count": len(profiles),
                "total_expected": total_expected,
                "is_generating": is_generating,
                "file_exists": file_exists,
                "file_modified_at": file_modified_at,
                "profiles": profiles
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get Profile in real-time: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/config/realtime')
async def get_simulation_config_realtime(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Real-time retrieval of simulation configurations (used to view progress in real-time during generation).
    """
    try:
        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.simulationNotFound', id=simulation_id)
                }
            )
        
        # Configuration file path
        config_file = os.path.join(sim_dir, "simulation_config.json")
        
        # Check if file exists
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None
        
        if file_exists:
            # Get file modification time
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                def read_config():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                config = await run_in_threadpool(read_config)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read config file (might be in write mode): {e}")
                config = None
        
        # Check if generation is in progress (judged via state.json)
        is_generating = False
        generation_stage = None
        config_generated = False
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                def read_state():
                    with open(state_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                state_data = await run_in_threadpool(read_state)
                status_val = state_data.get("status", "")
                is_generating = status_val == "preparing"
                config_generated = state_data.get("config_generated", False)
                
                # Determine current stage
                if is_generating:
                    if state_data.get("profiles_generated", False):
                        generation_stage = "generating_config"
                    else:
                        generation_stage = "generating_profiles"
                elif status_val == "ready":
                    generation_stage = "completed"
            except Exception:
                pass
        
        # Build response data
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config
        }
        
        # If configuration exists, extract some key statistical info
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get("total_simulation_hours"),
                "initial_posts_count": len(config.get("event_config", {}).get("initial_posts", [])),
                "hot_topics_count": len(config.get("event_config", {}).get("hot_topics", [])),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model")
            }
        
        return {
            "success": True,
            "data": response_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get Config in real-time: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
