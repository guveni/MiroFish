"""
Main Simulation API router that integrates CRUD, entities, realtime, actions, and social routes.
"""

from fastapi import APIRouter
from .crud import router as crud_router
from .entities import router as entities_router
from .realtime import router as realtime_router
from .actions import router as actions_router
from .social import router as social_router

router = APIRouter()

router.include_router(crud_router)
router.include_router(entities_router)
router.include_router(realtime_router)
router.include_router(actions_router)
router.include_router(social_router)
