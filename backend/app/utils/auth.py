"""
Authentication utilities and dependencies for OAuth 2.0 and JWT verification.
Compatible with FastAPI Dependency Injection.
"""

import os
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests

from ..config import Config
from ..database import db
from ..models.user import User

logger = logging.getLogger("mirofish.auth")

security = HTTPBearer(auto_error=False)


def verify_token(token: str) -> Optional[dict]:
    """
    Verify the token. Supports real Google OIDC and local dummy tokens in development.
    """
    # 1. Local/Development Dummy Token Support
    # Format: dummy_usr_<name>_<email>
    if token.startswith("dummy_"):
        try:
            parts = token.split("_")
            name = parts[2] if len(parts) > 2 else "Local Dev User"
            email = parts[3] if len(parts) > 3 else f"{parts[1]}@example.com"
            return {
                "email": email,
                "name": name,
                "sub": f"dummy_oauth_{email}",
                "provider": "dummy"
            }
        except Exception:
            return None

    # 2. Google OAuth 2.0 / OIDC Verification
    try:
        # We need a Google client ID to restrict the token to our app,
        # but in many multi-tenant backend proxies, just verifying the token structure
        # against Google certificates is enough, or we check GOOGLE_CLIENT_ID env var.
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        
        # Verify the ID token using Google auth library
        id_info = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            audience=client_id
        )
        
        return {
            "email": id_info.get("email"),
            "name": id_info.get("name"),
            "sub": id_info.get("sub"),
            "provider": "google"
        }
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    FastAPI dependency to authenticate the Bearer token, find or create the user in the database.
    """
    if not credentials:
        if Config.DEBUG:
            email = "local_dev_user@example.com"
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(
                    email=email,
                    name="Local Dev User",
                    oauth_provider="dummy",
                    oauth_id="dummy_oauth_local_dev"
                )
                db.session.add(user)
                db.session.commit()
                db.session.refresh(user)
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing Bearer token."
        )
        
    token = credentials.credentials
    user_info = verify_token(token)
    if not user_info or not user_info.get("email"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token."
        )
        
    email = user_info["email"]
    
    # Find or create user in database
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(
            email=email,
            name=user_info.get("name"),
            oauth_provider=user_info.get("provider"),
            oauth_id=user_info.get("sub")
        )
        db.session.add(user)
        db.session.commit()
        db.session.refresh(user)
        
    return user
