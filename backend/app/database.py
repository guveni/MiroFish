"""
Database module initializing SQLAlchemy with standard/vanilla SQLAlchemy.
Exposes a Flask-SQLAlchemy compatible interface to keep model definitions intact.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship

Base = declarative_base()


class SQLAlchemy:
    """Flask-SQLAlchemy compatible wrapper for vanilla SQLAlchemy."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._session = None
        self.Model = Base
        
        # Expose common SQLAlchemy elements for backward compatibility in models
        self.Column = Column
        self.Integer = Integer
        self.String = String
        self.DateTime = DateTime
        self.Text = Text
        self.JSON = JSON
        self.ForeignKey = ForeignKey
        self.relationship = relationship

    def init_app(self, app_or_url: str):
        """Initialize the database engine and session factories."""
        # Handle if an app object was passed (legacy Flask style) or a connection URL string
        if isinstance(app_or_url, str):
            database_url = app_or_url
        else:
            # app object passed, get database URI from config
            database_url = app_or_url.config.get("SQLALCHEMY_DATABASE_URI")
            
        connect_args = {}
        # Special handling for SQLite
        if database_url and database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args=connect_args
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._session = scoped_session(self.SessionLocal)
        
        # Attach .query to the Base class to support Model.query
        Base.query = self._session.query_property()

    @property
    def session(self):
        """Return the scoped session."""
        if self._session is None:
            raise RuntimeError("Database is not initialized. Call init_app first.")
        return self._session

    def create_all(self):
        """Create all tables in the database."""
        if self.engine is not None:
            Base.metadata.create_all(bind=self.engine)


db = SQLAlchemy()


def get_db():
    """FastAPI dependency to yield a request-scoped database session."""
    if db.SessionLocal is None:
        raise RuntimeError("Database is not initialized. Call db.init_app first.")
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()
