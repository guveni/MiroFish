"""
Main Report API router that integrates CRUD and chat/log/tool routes.
"""

from fastapi import APIRouter
from .crud import router as crud_router
from .chat import router as chat_router

router = APIRouter()

router.include_router(crud_router)
router.include_router(chat_router)
