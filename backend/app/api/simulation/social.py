"""
Simulation-related social and agent query API routes
"""

import os
import traceback
import sqlite3
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...services.simulation_runner import SimulationRunner
from ...utils.logger import get_logger
from ...utils.locale import t
from ...models.user import User
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.simulation.social')

# Interview prompt optimization prefix
INTERVIEW_PROMPT_PREFIX = "Based on your persona, all your past memories and actions, reply to me directly using text without calling any tools: "


def optimize_interview_prompt(prompt: str) -> str:
    """
    Optimize interview prompt, adding a prefix to avoid Agent calling tools.
    """
    if not prompt:
        return prompt
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


@router.get('/{simulation_id}/agent-stats')
async def get_agent_stats(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get statistical information of each Agent.
    """
    try:
        stats = await run_in_threadpool(SimulationRunner.get_agent_stats, simulation_id)
        
        return {
            "success": True,
            "data": {
                "agents_count": len(stats),
                "stats": stats
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get Agent statistics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/posts')
async def get_simulation_posts(
    simulation_id: str,
    platform: str = Query('reddit'),
    limit: int = Query(50),
    offset: int = Query(0),
    current_user: User = Depends(get_current_user)
):
    """
    Get posts in the simulation.
    """
    try:
        # Try fetching from high-throughput PostgreSQL first
        from ...utils.postgres_sync import get_simulation_posts_from_pg
        pg_result = await run_in_threadpool(
            get_simulation_posts_from_pg,
            simulation_id, platform, limit, offset
        )
        if pg_result is not None:
            posts, total = pg_result
            return {
                "success": True,
                "data": {
                    "platform": platform,
                    "total": total,
                    "count": len(posts),
                    "posts": posts
                }
            }

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../../uploads/simulations/{simulation_id}'
        )
        
        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)
        
        if not os.path.exists(db_path):
            return {
                "success": True,
                "data": {
                    "platform": platform,
                    "count": 0,
                    "posts": [],
                    "message": t('api.dbNotExist')
                }
            }
        
        def query_sqlite():
            db_abs_path = os.path.abspath(db_path)
            conn = sqlite3.connect(f"file:{db_abs_path}?mode=ro&nolock=1", uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT * FROM post 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                posts_list = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT COUNT(*) FROM post")
                total_count = cursor.fetchone()[0]
                
            except sqlite3.OperationalError:
                posts_list = []
                total_count = 0
            finally:
                conn.close()
            return posts_list, total_count

        posts, total = await run_in_threadpool(query_sqlite)
        
        return {
            "success": True,
            "data": {
                "platform": platform,
                "total": total,
                "count": len(posts),
                "posts": posts
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get posts: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{simulation_id}/comments')
async def get_simulation_comments(
    simulation_id: str,
    post_id: Optional[str] = Query(None),
    limit: int = Query(50),
    offset: int = Query(0),
    current_user: User = Depends(get_current_user)
):
    """
    Get comments in the simulation (Reddit only).
    """
    try:
        # Try fetching from high-throughput PostgreSQL first
        from ...utils.postgres_sync import get_simulation_comments_from_pg
        pg_result = await run_in_threadpool(
            get_simulation_comments_from_pg,
            simulation_id, post_id, limit, offset
        )
        if pg_result is not None:
            return {
                "success": True,
                "data": {
                    "count": len(pg_result),
                    "comments": pg_result
                }
            }

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../../uploads/simulations/{simulation_id}'
        )
        
        db_path = os.path.join(sim_dir, "reddit_simulation.db")
        
        if not os.path.exists(db_path):
            return {
                "success": True,
                "data": {
                    "count": 0,
                    "comments": []
                }
            }
        
        def query_sqlite():
            db_abs_path = os.path.abspath(db_path)
            conn = sqlite3.connect(f"file:{db_abs_path}?mode=ro&nolock=1", uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                if post_id:
                    cursor.execute("""
                        SELECT * FROM comment 
                        WHERE post_id = ?
                        ORDER BY created_at DESC 
                        LIMIT ? OFFSET ?
                    """, (post_id, limit, offset))
                else:
                    cursor.execute("""
                        SELECT * FROM comment 
                        ORDER BY created_at DESC 
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                
                comments_list = [dict(row) for row in cursor.fetchall()]
                
            except sqlite3.OperationalError:
                comments_list = []
            finally:
                conn.close()
            return comments_list

        comments = await run_in_threadpool(query_sqlite)
        
        return {
            "success": True,
            "data": {
                "count": len(comments),
                "comments": comments
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get comments: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/interview')
async def interview_agent(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Interview a single Agent.
    """
    try:
        simulation_id = data.get('simulation_id')
        agent_id = data.get('agent_id')
        prompt = data.get('prompt')
        platform = data.get('platform')
        timeout = data.get('timeout', 60)
        
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )
        
        if agent_id is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireAgentId')
                }
            )
        
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requirePrompt')
                }
            )
        
        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.invalidInterviewPlatform')
                }
            )
        
        # Check environment status
        env_alive = await run_in_threadpool(SimulationRunner.check_env_alive, simulation_id)
        if not env_alive:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.envNotRunning')
                }
            )
        
        optimized_prompt = optimize_interview_prompt(prompt)
        
        result = await run_in_threadpool(
            SimulationRunner.interview_agent,
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

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
        
    except TimeoutError as e:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": t('api.interviewTimeout', error=str(e))
            }
        )
        
    except Exception as e:
        logger.error(f"Interview failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/interview/batch')
async def interview_agents_batch(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Batch interview multiple Agents.
    """
    try:
        simulation_id = data.get('simulation_id')
        interviews = data.get('interviews')
        platform = data.get('platform')
        timeout = data.get('timeout', 120)

        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        if not interviews or not isinstance(interviews, list):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireInterviews')
                }
            )

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.invalidInterviewPlatform')
                }
            )

        # Validate each interview item
        for i, interview in enumerate(interviews):
            if 'agent_id' not in interview:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.interviewListMissingAgentId', index=i+1)
                    }
                )
            if 'prompt' not in interview:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.interviewListMissingPrompt', index=i+1)
                    }
                )
            item_platform = interview.get('platform')
            if item_platform and item_platform not in ("twitter", "reddit"):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": t('api.interviewListInvalidPlatform', index=i+1)
                    }
                )

        # Check environment status
        env_alive = await run_in_threadpool(SimulationRunner.check_env_alive, simulation_id)
        if not env_alive:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.envNotRunning')
                }
            )

        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview['prompt'] = optimize_interview_prompt(interview.get('prompt', ''))
            optimized_interviews.append(optimized_interview)

        result = await run_in_threadpool(
            SimulationRunner.interview_agents_batch,
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout
        )

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

    except TimeoutError as e:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": t('api.batchInterviewTimeout', error=str(e))
            }
        )

    except Exception as e:
        logger.error(f"Batch interview failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/interview/all')
async def interview_all_agents(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Global interview - Interview all Agents with the same question.
    """
    try:
        simulation_id = data.get('simulation_id')
        prompt = data.get('prompt')
        platform = data.get('platform')
        timeout = data.get('timeout', 180)

        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requirePrompt')
                }
            )

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.invalidInterviewPlatform')
                }
            )

        # Check environment status
        env_alive = await run_in_threadpool(SimulationRunner.check_env_alive, simulation_id)
        if not env_alive:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.envNotRunning')
                }
            )

        optimized_prompt = optimize_interview_prompt(prompt)

        result = await run_in_threadpool(
            SimulationRunner.interview_all_agents,
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

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

    except TimeoutError as e:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": t('api.globalInterviewTimeout', error=str(e))
            }
        )

    except Exception as e:
        logger.error(f"Global interview failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/interview/history')
async def get_interview_history(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Get interview history.
    """
    try:
        simulation_id = data.get('simulation_id')
        platform = data.get('platform')
        agent_id = data.get('agent_id')
        limit = data.get('limit', 100)
        
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        history = await run_in_threadpool(
            SimulationRunner.get_interview_history,
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit
        )

        return {
            "success": True,
            "data": {
                "count": len(history),
                "history": history
            }
        }

    except Exception as e:
        logger.error(f"Failed to get interview history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
