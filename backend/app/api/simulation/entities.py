"""
Entity reading API routes for simulation graphs.
"""

import traceback
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...services import _backend
from ...services.zep_entity_reader import ZepEntityReader
from ...utils.logger import get_logger
from ...utils.locale import t
from ...models.user import User
from ...utils.auth import get_current_user

router = APIRouter()
logger = get_logger('mirofish.api.simulation.entities')


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


@router.get('/entities/{graph_id}')
async def get_graph_entities(
    graph_id: str,
    entity_types: Optional[str] = Query(None),
    enrich: bool = Query(True),
    current_user: User = Depends(get_current_user)
):
    """
    Get all entities in the graph (filtered).
    """
    try:
        backend_error = _graph_backend_error_response()
        if backend_error:
            return backend_error
        
        parsed_entity_types = [t.strip() for t in entity_types.split(',') if t.strip()] if entity_types else None
        
        logger.info(f"Get graph entities: graph_id={graph_id}, entity_types={parsed_entity_types}, enrich={enrich}")
        
        reader = ZepEntityReader()
        result = await run_in_threadpool(
            reader.filter_defined_entities,
            graph_id=graph_id,
            defined_entity_types=parsed_entity_types,
            enrich_with_edges=enrich
        )
        
        return {
            "success": True,
            "data": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph entities: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/entities/{graph_id}/{entity_uuid}')
async def get_entity_detail(
    graph_id: str,
    entity_uuid: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed information of a single entity."""
    try:
        backend_error = _graph_backend_error_response()
        if backend_error:
            return backend_error
        
        reader = ZepEntityReader()
        entity = await run_in_threadpool(reader.get_entity_with_context, graph_id, entity_uuid)
        
        if not entity:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": t('api.entityNotFound', id=entity_uuid)
                }
            )
        
        return {
            "success": True,
            "data": entity.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get entity details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@router.get('/entities/{graph_id}/by-type/{entity_type}')
async def get_entities_by_type(
    graph_id: str,
    entity_type: str,
    enrich: bool = Query(True),
    current_user: User = Depends(get_current_user)
):
    """Get all entities of a specified type."""
    try:
        backend_error = _graph_backend_error_response()
        if backend_error:
            return backend_error
        
        reader = ZepEntityReader()
        entities = await run_in_threadpool(
            reader.get_entities_by_type,
            graph_id=graph_id,
            entity_type=entity_type,
            enrich_with_edges=enrich
        )
        
        return {
            "success": True,
            "data": {
                "entity_type": entity_type,
                "count": len(entities),
                "entities": [e.to_dict() for e in entities]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get entities: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
