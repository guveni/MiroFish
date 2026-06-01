"""
User model for multi-tenant database authentication and isolation.
"""

import uuid
from datetime import datetime
from ..database import db

class User(db.Model):
    """User data model."""
    __tablename__ = 'users'
    
    id = db.Column(db.String(50), primary_key=True, default=lambda: f"usr_{uuid.uuid4().hex[:12]}")
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=True)
    oauth_provider = db.Column(db.String(50), nullable=True)  # e.g. 'google', 'github'
    oauth_id = db.Column(db.String(255), nullable=True)  # Provider ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    projects = db.relationship('Project', backref='user_rel', lazy=True, cascade="all, delete-orphan")
    tasks = db.relationship('Task', backref='user_rel', lazy=True, cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert user to a dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "oauth_provider": self.oauth_provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
