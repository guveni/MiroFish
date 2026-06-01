"""
Report-related CRUD and metadata API routes
"""

import os
import traceback
from typing import Optional
from fastapi import APIRouter, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool

from ...services.report_agent import ReportManager, ReportStatus
from ...services.simulation_manager import SimulationManager
from ...models.project import ProjectManager
from ...models.task import TaskManager
from ...models.user import User
from ...utils.logger import get_logger
from ...utils.locale import t, get_locale
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.report.crud')


# ============== Report Generation Interface ==============

@router.post('/generate')
async def generate_report(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Generate simulation analysis report (Asynchronous task).
    
    This is a time-consuming operation. The endpoint returns a task_id immediately.
    Use GET /api/report/generate/status to query progress.
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",    // Required, simulation ID
            "force_regenerate": false        // Optional, force regeneration
        }
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "Report generation task started"
            }
        }
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

        force_regenerate = data.get('force_regenerate', False)
        
        # Get simulation info
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

        # Get project info
        project = await run_in_threadpool(ProjectManager.get_project, state.project_id)
        if not project or (project.user_id and project.user_id != current_user.id):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.projectNotFound', id=state.project_id)
                }
            )

        # Check if report already exists
        if not force_regenerate:
            existing_report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "message": t('api.reportAlreadyExists'),
                        "already_generated": True
                    }
                }
        else:
            # Clean up old report to prevent conflicts
            existing_report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
            if existing_report:
                logger.info(f"Deleting existing report {existing_report.report_id} for simulation {simulation_id} because of force_regenerate")
                await run_in_threadpool(ReportManager.delete_report, existing_report.report_id)
        
        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.missingGraphIdEnsure')
                }
            )
        
        simulation_requirement = project.simulation_requirement
        instructions = data.get('instructions')
        if instructions:
            simulation_requirement += f"\\n\\nAdditional instructions for report generation: {instructions}"
            
        if not simulation_requirement:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": t('api.missingSimRequirement')
                }
            )
        
        # Get/generate report_id in advance so it can be returned immediately to the frontend
        report_id = data.get('report_id')
        if not report_id:
            if not force_regenerate:
                existing_report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
                if existing_report:
                    report_id = existing_report.report_id
            
            if not report_id:
                import uuid
                report_id = f"report_{uuid.uuid4().hex[:12]}"
                
        # Synchronously initialize Report metadata to make sure frontend can query it
        from datetime import datetime
        initial_report = await run_in_threadpool(ReportManager.get_report, report_id)
        if not initial_report:
            from ...services.report_agent import Report
            initial_report = Report(
                report_id=report_id,
                simulation_id=simulation_id,
                graph_id=graph_id,
                simulation_requirement=simulation_requirement,
                status=ReportStatus.PENDING,
                created_at=datetime.now().isoformat()
            )
            await run_in_threadpool(ReportManager.save_report, initial_report)
        
        # Create background/asynchronous task
        task_manager = TaskManager()
        task_id = await run_in_threadpool(
            task_manager.create_task,
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id
            },
            user_id=current_user.id
        )
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Send event to Inngest for background execution
        from ...inngest_client import inngest_client
        import inngest
        
        event = inngest.Event(
            name="report/generate",
            data={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id,
                "simulation_requirement": simulation_requirement,
                "task_id": task_id,
                "locale": current_locale,
            }
        )
        await run_in_threadpool(inngest_client.send_sync, event)
        
        return {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "report_id": report_id,
                "task_id": task_id,
                "status": "generating",
                "message": t('api.reportGenerateStarted'),
                "already_generated": False
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to start report generation task: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.post('/generate/status')
async def get_generate_status(
    data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Query progress of the report generation task.
    
    Request (JSON):
        {
            "task_id": "task_xxxx",         // Optional, task_id returned by generate
            "simulation_id": "sim_xxxx"     // Optional, simulation ID
        }
    
    Response:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|failed",
                "progress": 45,
                "message": "..."
            }
        }
    """
    try:
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # If simulation_id is provided, check if a completed report already exists
        if simulation_id:
            existing_report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "progress": 100,
                        "message": t('api.reportGenerated'),
                        "already_completed": True
                    }
                }
        
        if not task_id:
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
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.taskNotFound', id=task_id)
                }
            )
        
        return {
            "success": True,
            "data": task.to_dict()
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


# ============== Report Retrieval Interfaces ==============

@router.get('/{report_id}')
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get report details.
    
    Response:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "simulation_id": "sim_xxxx",
                "status": "completed",
                "outline": {...},
                "markdown_content": "...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        report = await run_in_threadpool(ReportManager.get_report, report_id)
        
        if not report:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.reportNotFound', id=report_id)
                }
            )
        
        return {
            "success": True,
            "data": report.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get report by ID: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/by-simulation/{simulation_id}')
async def get_report_by_simulation(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get report by simulation ID.
    
    Response:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                ...
            }
        }
    """
    try:
        report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
        
        if not report:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.noReportForSim', id=simulation_id),
                    "has_report": False
                }
            )
        
        return {
            "success": True,
            "data": report.to_dict(),
            "has_report": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get report by simulation ID: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/list')
async def list_reports(
    simulation_id: Optional[str] = Query(None),
    limit: int = Query(50),
    current_user: User = Depends(get_current_user)
):
    """
    List all reports.
    
    Query Parameters:
        simulation_id: Filter by simulation ID (optional)
        limit: Return count limit (default 50)
    
    Response:
        {
            "success": true,
            "data": [...],
            "count": 10
        }
    """
    try:
        reports = await run_in_threadpool(
            ReportManager.list_reports,
            simulation_id=simulation_id,
            limit=limit
        )
        
        return {
            "success": True,
            "data": [r.to_dict() for r in reports],
            "count": len(reports)
        }
        
    except Exception as e:
        logger.error(f"Failed to list reports: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{report_id}/download')
async def download_report(
    report_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Download report (Markdown format).
    
    Returns the Markdown file.
    """
    try:
        report = await run_in_threadpool(ReportManager.get_report, report_id)
        
        if not report:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.reportNotFound', id=report_id)
                }
            )
        
        md_path = ReportManager._get_report_markdown_path(report_id)
        
        if not os.path.exists(md_path):
            # If the MD file does not exist, generate a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(report.markdown_content)
                temp_path = f.name
            
            background_tasks.add_task(os.unlink, temp_path)
            return FileResponse(
                temp_path,
                filename=f"{report_id}.md",
                media_type="text/markdown"
            )
        
        return FileResponse(
            md_path,
            filename=f"{report_id}.md",
            media_type="text/markdown"
        )
        
    except Exception as e:
        logger.error(f"Failed to download report: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.delete('/{report_id}')
async def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete report."""
    try:
        success = await run_in_threadpool(ReportManager.delete_report, report_id)
        
        if not success:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.reportNotFound', id=report_id)
                }
            )
        
        return {
            "success": True,
            "message": t('api.reportDeleted', id=report_id)
        }
        
    except Exception as e:
        logger.error(f"Failed to delete report: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# ============== Report Progress and Chunked Generation Interfaces ==============

@router.get('/{report_id}/progress')
async def get_report_progress(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get report generation progress (real-time).
    
    Response:
        {
            "success": true,
            "data": {
                "status": "generating",
                "progress": 45,
                "message": "Generating section: Key Findings",
                "current_section": "Key Findings",
                "completed_sections": ["Executive Summary", "Simulation Background"],
                "updated_at": "2025-12-09T..."
            }
        }
    """
    try:
        progress = await run_in_threadpool(ReportManager.get_progress, report_id)
        
        if not progress:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.reportProgressNotAvail', id=report_id)
                }
            )
        
        return {
            "success": True,
            "data": progress
        }
        
    except Exception as e:
        logger.error(f"Failed to get report progress: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{report_id}/sections')
async def get_report_sections(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the list of generated sections (chunked output).
    
    The frontend can poll this endpoint to retrieve already generated section contents without waiting for the entire report to complete.
    
    Response:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "sections": [
                    {
                        "filename": "section_01.md",
                        "section_index": 1,
                        "content": "## Executive Summary\\n\\n..."
                    },
                    ...
                ],
                "total_sections": 3,
                "is_complete": false
            }
        }
    """
    try:
        sections = await run_in_threadpool(ReportManager.get_generated_sections, report_id)
        
        # Get report status
        report = await run_in_threadpool(ReportManager.get_report, report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED
        
        return {
            "success": True,
            "data": {
                "report_id": report_id,
                "sections": sections,
                "total_sections": len(sections),
                "is_complete": is_complete
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get section list: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/{report_id}/section/{section_index}')
async def get_single_section(
    report_id: str,
    section_index: int,
    current_user: User = Depends(get_current_user)
):
    """
    Get the content of a single section.
    
    Response:
        {
            "success": true,
            "data": {
                "filename": "section_01.md",
                "content": "## Executive Summary\\n\\n..."
            }
        }
    """
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)
        
        if not os.path.exists(section_path):
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.sectionNotFound', index=f"{section_index:02d}")
                }
            )
        
        def _read_file(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
                
        content = await run_in_threadpool(_read_file, section_path)
        
        return {
            "success": True,
            "data": {
                "filename": f"section_{section_index:02d}.md",
                "section_index": section_index,
                "content": content
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get section content: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# ============== Report Status Check Interface ==============

@router.get('/check/{simulation_id}')
async def check_report_status(
    simulation_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check if a simulation has a report and check the report status.
    
    Used by the frontend to determine whether to unlock the Interview feature.
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "has_report": true,
                "report_status": "completed",
                "report_id": "report_xxxx",
                "interview_unlocked": true
            }
        }
    """
    try:
        report = await run_in_threadpool(ReportManager.get_report_by_simulation, simulation_id)
        
        has_report = report is not None
        report_status = report.status.value if report else None
        report_id = report.report_id if report else None
        
        # Only unlock interview if the report is completed
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED
        
        return {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "has_report": has_report,
                "report_status": report_status,
                "report_id": report_id,
                "interview_unlocked": interview_unlocked
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to check report status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
