"""
Project context management using a SQL database backend.
Persists project state server-side so the frontend does not need to pass large
payloads between API calls.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from ..config import Config
from ..database import db


class ProjectStatus(str, Enum):
    """Project status."""
    CREATED = "created"              # Created, files uploaded
    ONTOLOGY_GENERATED = "ontology_generated"  # Ontology generated
    GRAPH_BUILDING = "graph_building"    # Graph build in progress
    GRAPH_COMPLETED = "graph_completed"  # Graph build completed
    FAILED = "failed"                # Failed


class Project(db.Model):
    """Project data model stored in database."""
    __tablename__ = 'projects'

    project_id = db.Column(db.String(50), primary_key=True)
    user_id = db.Column(db.String(50), db.ForeignKey('users.id'), nullable=True)
    name = db.Column(db.String(255), nullable=False, default="Unnamed Project")
    status = db.Column(db.String(50), nullable=False, default="created")
    created_at = db.Column(db.String(100), nullable=False)
    updated_at = db.Column(db.String(100), nullable=False)

    # JSON fields
    files = db.Column(db.JSON, nullable=False, default=list)  # [{'filename', 'path', 'size'}]
    total_text_length = db.Column(db.Integer, default=0)

    ontology = db.Column(db.JSON, nullable=True)  # {'entity_types', 'edge_types'}
    analysis_summary = db.Column(db.Text, nullable=True)

    graph_id = db.Column(db.String(100), nullable=True)
    graph_build_task_id = db.Column(db.String(100), nullable=True)
    graph_build_progress = db.Column(db.Integer, default=0)
    graph_build_message = db.Column(db.Text, nullable=True)
    ontology_task_id = db.Column(db.String(100), nullable=True)

    simulation_requirement = db.Column(db.Text, nullable=True)
    chunk_size = db.Column(db.Integer, default=500)
    chunk_overlap = db.Column(db.Integer, default=50)

    gemini_grounding_metadata = db.Column(db.JSON, nullable=True)
    error = db.Column(db.Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "name": self.name,
            "status": self.status.value if isinstance(self.status, ProjectStatus) else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": self.files,
            "total_text_length": self.total_text_length,
            "ontology": self.ontology,
            "analysis_summary": self.analysis_summary,
            "graph_id": self.graph_id,
            "graph_build_task_id": self.graph_build_task_id,
            "graph_build_progress": self.graph_build_progress,
            "graph_build_message": self.graph_build_message,
            "ontology_task_id": self.ontology_task_id,
            "simulation_requirement": self.simulation_requirement,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "gemini_grounding_metadata": self.gemini_grounding_metadata,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create a transient object from a dictionary (useful for fallback / testing)."""
        status = data.get('status', 'created')
        if isinstance(status, str):
            status = ProjectStatus(status)

        return cls(
            project_id=data['project_id'],
            user_id=data.get('user_id'),
            name=data.get('name', 'Unnamed Project'),
            status=status.value if isinstance(status, ProjectStatus) else status,
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            files=data.get('files', []),
            total_text_length=data.get('total_text_length', 0),
            ontology=data.get('ontology'),
            analysis_summary=data.get('analysis_summary'),
            graph_id=data.get('graph_id'),
            graph_build_task_id=data.get('graph_build_task_id'),
            graph_build_progress=int(data.get('graph_build_progress') or 0),
            graph_build_message=data.get('graph_build_message') or "",
            ontology_task_id=data.get('ontology_task_id'),
            simulation_requirement=data.get('simulation_requirement'),
            chunk_size=data.get('chunk_size', 500),
            chunk_overlap=data.get('chunk_overlap', 50),
            gemini_grounding_metadata=data.get('gemini_grounding_metadata'),
            error=data.get('error')
        )


class ProjectManager:
    """Project manager for persistent database storage and retrieval."""

    PROJECTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'projects')

    @classmethod
    def _ensure_projects_dir(cls):
        """Ensure the project directory exists (for temporary file storage)."""
        os.makedirs(cls.PROJECTS_DIR, exist_ok=True)

    @classmethod
    def _get_project_dir(cls, project_id: str) -> str:
        """Return the project directory path."""
        return os.path.join(cls.PROJECTS_DIR, project_id)

    @classmethod
    def _get_project_files_dir(cls, project_id: str) -> str:
        """Return the project file storage directory."""
        return os.path.join(cls._get_project_dir(project_id), 'files')

    @classmethod
    def _get_project_text_path(cls, project_id: str) -> str:
        """Return the extracted text storage path."""
        return os.path.join(cls._get_project_dir(project_id), 'extracted_text.txt')

    @classmethod
    def _get_legacy_project_json_path(cls, project_id: str) -> str:
        """Return the legacy on-disk project.json path (pre-PostgreSQL migrations)."""
        return os.path.join(cls._get_project_dir(project_id), 'project.json')

    @classmethod
    def get_display_files(cls, project: Project, limit: int = 3) -> List[Dict[str, str]]:
        """Return file metadata for UI display, falling back to on-disk legacy data."""
        db_files = project.files or []
        if db_files:
            return [
                {"filename": f.get("filename") or f.get("original_filename") or "Unknown File"}
                for f in db_files[:limit]
            ]

        legacy_path = cls._get_legacy_project_json_path(project.project_id)
        if os.path.isfile(legacy_path):
            try:
                with open(legacy_path, 'r', encoding='utf-8') as f:
                    legacy = json.load(f)
                legacy_files = legacy.get("files") or []
                if legacy_files:
                    return [
                        {"filename": f.get("filename") or f.get("original_filename") or "Unknown File"}
                        for f in legacy_files[:limit]
                    ]
            except Exception:
                pass

        disk_files = cls.get_project_files(project.project_id)
        return [{"filename": os.path.basename(path)} for path in disk_files[:limit]]

    @classmethod
    def _ensure_project_dir(cls, project_id: str) -> str:
        """Ensure the on-disk project directory exists (legacy projects may lack it)."""
        cls._ensure_projects_dir()
        project_dir = cls._get_project_dir(project_id)
        files_dir = cls._get_project_files_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(files_dir, exist_ok=True)
        return project_dir

    @classmethod
    def create_project(cls, name: str = "Unnamed Project", user_id: Optional[str] = None) -> Project:
        """Create a new project in the database."""
        cls._ensure_projects_dir()

        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        project = Project(
            project_id=project_id,
            user_id=user_id,
            name=name,
            status=ProjectStatus.CREATED.value,
            created_at=now,
            updated_at=now,
            files=[]
        )

        # Create the local project directory structure for temporary files.
        cls._ensure_project_dir(project_id)

        db.session.add(project)
        db.session.commit()

        return project

    @classmethod
    def save_project(cls, project: Project) -> None:
        """Save project metadata to the database."""
        project.updated_at = datetime.now().isoformat()
        
        # Ensure the project is merged into the session if it's detached
        if project not in db.session:
            db.session.merge(project)
        db.session.commit()

    @classmethod
    def get_project(cls, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return db.session.get(Project, project_id)

    @classmethod
    def list_projects(cls, user_id: Optional[str] = None, limit: int = 50) -> List[Project]:
        """List all projects, optionally filtered by user_id."""
        query = Project.query
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        # Sort by created_at descending.
        projects = query.order_by(Project.created_at.desc()).limit(limit).all()
        return projects

    @classmethod
    def delete_project(cls, project_id: str) -> bool:
        """Delete a project and all of its files."""
        project = cls.get_project(project_id)
        if not project:
            return False

        db.session.delete(project)
        db.session.commit()

        # Clean up GCS blobs if configured
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if bucket_name:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=f"projects/{project_id}/")
                for blob in blobs:
                    blob.delete()
            except Exception as e:
                # Log but do not block deletion
                pass
        
        # Clean up local files
        project_dir = cls._get_project_dir(project_id)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        return True

    @classmethod
    def save_file_to_project(cls, project_id: str, file_storage, original_filename: str) -> Dict[str, str]:
        """Save an uploaded file to the project directory or GCS."""
        cls._ensure_projects_dir()

        # Generate a safe filename.
        ext = os.path.splitext(original_filename)[1].lower()
        safe_filename = f"{uuid.uuid4().hex[:8]}{ext}"
        
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if bucket_name:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"projects/{project_id}/files/{safe_filename}"
            blob = bucket.blob(blob_path)
            
            # Reset file stream and upload
            file_storage.file.seek(0)
            blob.upload_from_file(file_storage.file)
            
            file_size = blob.size or 0
            
            return {
                "original_filename": original_filename,
                "saved_filename": safe_filename,
                "path": blob_path,
                "size": file_size
            }
        else:
            files_dir = cls._get_project_files_dir(project_id)
            cls._ensure_project_dir(project_id)
            file_path = os.path.join(files_dir, safe_filename)
            
            # Save the file.
            import shutil
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_storage.file, buffer)
            file_size = os.path.getsize(file_path)

            return {
                "original_filename": original_filename,
                "saved_filename": safe_filename,
                "path": file_path,
                "size": file_size
            }

    @classmethod
    def save_extracted_text(cls, project_id: str, text: str) -> None:
        """Save extracted text."""
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if bucket_name:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"projects/{project_id}/extracted_text.txt"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(text, content_type="text/plain")
        else:
            cls._ensure_project_dir(project_id)
            text_path = cls._get_project_text_path(project_id)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)

    @classmethod
    def get_extracted_text(cls, project_id: str) -> Optional[str]:
        """Get extracted text."""
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if bucket_name:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"projects/{project_id}/extracted_text.txt"
            blob = bucket.blob(blob_path)
            if blob.exists():
                return blob.download_as_string().decode('utf-8')
            return None
        else:
            text_path = cls._get_project_text_path(project_id)

            if not os.path.exists(text_path):
                return None

            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read()

    @classmethod
    def get_project_files(cls, project_id: str) -> List[str]:
        """Return local file paths for a project. Downloads from GCS if needed."""
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if bucket_name:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create a temporary local folder
            local_temp_dir = os.path.join(Config.UPLOAD_FOLDER, 'temp', project_id)
            os.makedirs(local_temp_dir, exist_ok=True)
            
            blobs = bucket.list_blobs(prefix=f"projects/{project_id}/files/")
            local_paths = []
            for blob in blobs:
                filename = os.path.basename(blob.name)
                if not filename:
                    continue
                local_path = os.path.join(local_temp_dir, filename)
                blob.download_to_filename(local_path)
                local_paths.append(local_path)
                
            return local_paths
        else:
            files_dir = cls._get_project_files_dir(project_id)

            if not os.path.exists(files_dir):
                return []

            return [
                os.path.join(files_dir, f)
                for f in os.listdir(files_dir)
                if os.path.isfile(os.path.join(files_dir, f))
            ]
