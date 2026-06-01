"""
Task status management using a SQL database backend.
Tracks long-running tasks such as graph builds across multiple worker processes.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from ..database import db
from ..utils.locale import t


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"          # Waiting
    PROCESSING = "processing"    # Processing
    COMPLETED = "completed"      # Completed
    FAILED = "failed"            # Failed


class Task(db.Model):
    """Task data model stored in database."""
    __tablename__ = 'tasks'

    task_id = db.Column(db.String(50), primary_key=True)
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=True)
    task_type = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(50), nullable=False, default="pending")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    progress = db.Column(db.Integer, default=0)              # Overall progress percentage 0-100
    message = db.Column(db.Text, nullable=True, default="")  # Status message
    
    # JSON results/data
    result = db.Column(db.JSON, nullable=True)  # Task result
    error = db.Column(db.Text, nullable=True)    # Error information
    metadata_json = db.Column(db.JSON, nullable=True, default=dict)  # Extra metadata
    progress_detail = db.Column(db.JSON, nullable=True, default=dict)  # Detailed progress information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata_json,
        }


class TaskManager:
    """
    Task manager interface.
    Database-backed task status management.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern for compatibility."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def create_task(self, task_type: str, metadata: Optional[Dict] = None, user_id: Optional[str] = None) -> str:
        """Create a new task in the database."""
        task_id = str(uuid.uuid4())
        now = datetime.utcnow()

        task = Task(
            task_id=task_id,
            user_id=user_id,
            task_type=task_type,
            status=TaskStatus.PENDING.value,
            created_at=now,
            updated_at=now,
            metadata_json=metadata or {}
        )

        db.session.add(task)
        db.session.commit()

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return db.session.get(Task, task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """Update task status in the database."""
        task = self.get_task(task_id)
        if task:
            task.updated_at = datetime.utcnow()
            if status is not None:
                task.status = status.value if isinstance(status, TaskStatus) else status
            if progress is not None:
                task.progress = max(task.progress, progress)
            if message is not None:
                task.message = message
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if progress_detail is not None:
                task.progress_detail = progress_detail
            
            db.session.commit()

    def complete_task(self, task_id: str, result: Dict):
        """Mark a task as completed."""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message=t('progress.taskComplete'),
            result=result
        )

    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=t('progress.taskFailed'),
            error=error
        )

    def list_tasks(self, task_type: Optional[str] = None, user_id: Optional[str] = None) -> list:
        """List tasks from the database."""
        query = Task.query
        if task_type:
            query = query.filter_by(task_type=task_type)
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        tasks = query.order_by(Task.created_at.desc()).all()
        return [t.to_dict() for t in tasks]

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old tasks from the database."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        Task.query.filter(
            Task.created_at < cutoff,
            Task.status.in_([TaskStatus.COMPLETED.value, TaskStatus.FAILED.value])
        ).delete(synchronize_session=False)
        db.session.commit()
