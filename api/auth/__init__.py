"""Authentication API endpoints.

Implements SRS 2.1.2 / 2.1.3:
- POST /auth/login
- POST /auth/register
- POST /auth/logout
- GET  /auth/me
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.session import get_db
from db.schemas.user.auth_schema import (
    LoginRequest,
    RegisterRequest,
    AuthResponse,
    UserResponse,
)
from db.services.user.auth_service import (
    authenticate_user,
    create_user,
    create_access_token,
)
from api.deps import get_current_user
from db.models.user.user import UserModel

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=AuthResponse, summary="Login")
def login_api(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> AuthResponse:
    """Authenticate with username + password and receive a JWT."""
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(str(user.id))
    return AuthResponse(
        user=UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            created_at=user.created_at,
        ),
        access_token=token,
    )


@router.post("/register", response_model=AuthResponse, summary="Register")
def register_api(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> AuthResponse:
    """Create a new account and return a JWT."""
    try:
        user = create_user(db, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    token = create_access_token(str(user.id))
    return AuthResponse(
        user=UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            created_at=user.created_at,
        ),
        access_token=token,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Logout")
def logout_api() -> None:
    """Logout (client-side token removal). Server is stateless with JWT."""
    return None


@router.get("/me", response_model=UserResponse, summary="Current user")
def me_api(
    current_user: UserModel = Depends(get_current_user),
) -> UserResponse:
    """Return the currently authenticated user."""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )
