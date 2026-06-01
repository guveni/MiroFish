"""
Report-related chat, log, and tool API routes
"""

import traceback
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...services.report_agent import ReportAgent, ReportManager
from ...services.simulation_manager import SimulationManager
from ...models.project import ProjectManager
from ...models.user import User
from ...utils.logger import get_logger
from ...utils.locale import t
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.report.chat')


# ============== Report Agent Chat Interface ==============

@router.post('/chat')
async def chat_with_report_agent(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Chat with the Report Agent.
    
    The Report Agent can autonomously call retrieval tools during the conversation to answer questions.
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",        // Required, simulation ID
            "message": "Please explain the trend of public opinion",    // Required, user message
            "chat_history": [                   // Optional, chat history
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    
    Response:
        {
            "success": true,
            "data": {
                "response": "Agent response...",
                "tool_calls": [List of tools called],
                "sources": [Information sources]
            }
        }
    """
    try:
        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])
        
        if not simulation_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireSimulationId')
                }
            )

        if not message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireMessage')
                }
            )
        
        # Get simulation and project information
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

        project = await run_in_threadpool(ProjectManager.get_project, state.project_id)
        if not project:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.projectNotFound', id=state.project_id)
                }
            )
        
        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.missingGraphId')
                }
            )
        
        simulation_requirement = project.simulation_requirement or ""
        
        # Create Agent and chat
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement
        )
        
        result = await run_in_threadpool(agent.chat, message=message, chat_history=chat_history)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# ============== Agent Log Interface ==============

@router.get('/{report_id}/agent-log')
async def get_agent_log(
    report_id: str,
    from_line: int = Query(0),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed execution logs of the Report Agent.
    
    Get every action in real time during the report generation process, including:
    - Report start, planning start/completion
    - Start of each section, tool calls, LLM responses, completion
    - Report completion or failure
    
    Query Parameters:
        from_line: Line number to start reading from (optional, default 0, used for incremental fetching)
    
    Response:
        {
            "success": true,
            "data": {
                "logs": [
                    {
                        "timestamp": "2025-12-13T...",
                        "elapsed_seconds": 12.5,
                        "report_id": "report_xxxx",
                        "action": "tool_call",
                        "stage": "generating",
                        "section_title": "Executive Summary",
                        "section_index": 1,
                        "details": {
                            "tool_name": "insight_forge",
                            "parameters": {...},
                            ...
                        }
                    },
                    ...
                ],
                "total_lines": 25,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        log_data = await run_in_threadpool(ReportManager.get_agent_log, report_id, from_line=from_line)
        
        return {
            "success": True,
            "data": log_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get Agent log: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{report_id}/agent-log/stream')
async def stream_agent_log(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get full Agent logs (retrieve all at once).
    
    Response:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 25
            }
        }
    """
    try:
        logs = await run_in_threadpool(ReportManager.get_agent_log_stream, report_id)
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get Agent log stream: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.websocket('/{report_id}/agent-log/ws')
async def websocket_agent_log(websocket: WebSocket, report_id: str):
    """
    WebSocket endpoint for real-time streaming of Report Agent structured logs.
    """
    await websocket.accept()
    import asyncio
    
    from_line = 0
    try:
        while True:
            log_data = await run_in_threadpool(ReportManager.get_agent_log, report_id, from_line=from_line)
            logs = log_data.get("logs", [])
            if logs:
                await websocket.send_json({
                    "success": True,
                    "data": log_data
                })
                from_line = log_data.get("total_lines", from_line)
            
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info(f"WebSocket agent log disconnected for report {report_id}")
    except Exception as e:
        logger.error(f"WebSocket agent log error: {e}")


# ============== Console Log Interface ==============

@router.get('/{report_id}/console-log')
async def get_console_log(
    report_id: str,
    from_line: int = Query(0),
    current_user: User = Depends(get_current_user)
):
    """
    Get console output logs of the Report Agent.
    
    Get real-time console output (INFO, WARNING, etc.) during report generation.
    Unlike the agent-log interface which returns structured JSON,
    this endpoint provides plain text, console-style logs.
    
    Query Parameters:
        from_line: Line number to start reading from (optional, default 0, used for incremental fetching)
    
    Response:
        {
            "success": true,
            "data": {
                "logs": [
                    "[19:46:14] INFO: Search complete: Found 15 relevant facts",
                    "[19:46:14] INFO: Graph search: graph_id=xxx, query=...",
                    ...
                ],
                "total_lines": 100,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        log_data = await run_in_threadpool(ReportManager.get_console_log, report_id, from_line=from_line)
        
        return {
            "success": True,
            "data": log_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get console log: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{report_id}/console-log/stream')
async def stream_console_log(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get full console logs (retrieve all at once).
    
    Response:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 100
            }
        }
    """
    try:
        logs = await run_in_threadpool(ReportManager.get_console_log_stream, report_id)
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get console log stream: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.websocket('/{report_id}/console-log/ws')
async def websocket_console_log(websocket: WebSocket, report_id: str):
    """
    WebSocket endpoint for real-time streaming of Report Agent plain-text console logs.
    """
    await websocket.accept()
    import asyncio
    
    from_line = 0
    try:
        while True:
            log_data = await run_in_threadpool(ReportManager.get_console_log, report_id, from_line=from_line)
            logs = log_data.get("logs", [])
            if logs:
                await websocket.send_json({
                    "success": True,
                    "data": log_data
                })
                from_line = log_data.get("total_lines", from_line)
            
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info(f"WebSocket console log disconnected for report {report_id}")
    except Exception as e:
        logger.error(f"WebSocket console log error: {e}")


# ============== Tool Invocation Interfaces (for debugging) ==============

@router.post('/tools/search')
async def search_graph_tool(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Graph Search Tool Interface (for debugging)
    
    Request (JSON):
        {
            "graph_id": "mirofish_xxxx",
            "query": "Search query",
            "limit": 10
        }
    """
    try:
        graph_id = data.get('graph_id')
        query = data.get('query')
        limit = data.get('limit', 10)
        
        if not graph_id or not query:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.requireGraphIdAndQuery')
                }
            )
        
        from ...services.zep_tools import ZepToolsService
        
        tools = ZepToolsService()
        result = await run_in_threadpool(
            tools.search_graph,
            graph_id=graph_id,
            query=query,
            limit=limit
        )
        
        return {
            "success": True,
            "data": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Graph search failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/tools/statistics')
async def get_graph_statistics_tool(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Graph Statistics Tool Interface (for debugging)
    
    Request (JSON):
        {
            "graph_id": "mirofish_xxxx"
        }
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
        
        from ...services.zep_tools import ZepToolsService
        
        tools = ZepToolsService()
        result = await run_in_threadpool(tools.get_graph_statistics, graph_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
