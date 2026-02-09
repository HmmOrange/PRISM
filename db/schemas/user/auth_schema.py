"""Authentication request/response schemas.

Follows CODING_STANDARDS.md — separate models for request vs. response,
Pydantic v2 style, typed, with docstrings.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Request body for POST /auth/login."""

    username: str = Field(..., min_length=1, max_length=150)
    password: str = Field(..., min_length=4)


class RegisterRequest(BaseModel):
    """Request body for POST /auth/register."""

    username: str = Field(..., min_length=3, max_length=150)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """Public user representation returned in API responses."""

    id: str
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """Response returned after successful login / register."""

    user: UserResponse
    access_token: str
    token_type: str = "bearer"
