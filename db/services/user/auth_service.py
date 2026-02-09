"""Authentication service layer.

Implements SRS 2.1.3:
- bcrypt password hashing (per CODING_STANDARDS.md Security Guidelines)
- JWT token generation
- User CRUD helpers

Uses bcrypt directly for hashing and python-jose for JWT.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from db.models.user.user import UserModel
from db.schemas.user.auth_schema import RegisterRequest

# ---------------------------------------------------------------------------
# Password hashing — bcrypt with 12 rounds (per CODING_STANDARDS.md)
# ---------------------------------------------------------------------------

BCRYPT_ROUNDS = 12


def _truncate_password(password: str) -> bytes:
    """Encode and truncate password to 72 bytes (bcrypt limit)."""
    password_bytes = password.encode("utf-8")
    return password_bytes[:72]


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt."""
    password_bytes = _truncate_password(password)
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    try:
        plain_bytes = _truncate_password(plain)
        hashed_bytes = hashed.encode("utf-8")
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

JWT_SECRET_KEY = "prism-secret-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT for the given user."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Return the user_id (``sub`` claim) or ``None`` if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def get_user_by_username(db: Session, username: str) -> Optional[UserModel]:
    """Fetch a user by username (case-insensitive)."""
    return (
        db.query(UserModel)
        .filter(UserModel.username == username)
        .first()
    )


def get_user_by_email(db: Session, email: str) -> Optional[UserModel]:
    """Fetch a user by email."""
    return (
        db.query(UserModel)
        .filter(UserModel.email == email)
        .first()
    )


def get_user_by_id(db: Session, user_id: str) -> Optional[UserModel]:
    """Fetch a user by primary key."""
    return (
        db.query(UserModel)
        .filter(UserModel.id == user_id)
        .first()
    )


def create_user(db: Session, payload: RegisterRequest) -> UserModel:
    """Insert a new user row with a hashed password.

    Raises ``ValueError`` if username or email already taken.
    """
    if get_user_by_username(db, payload.username):
        raise ValueError("Username already taken")
    if get_user_by_email(db, payload.email):
        raise ValueError("Email already registered")

    user = UserModel(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(
    db: Session,
    username: str,
    password: str,
) -> Optional[UserModel]:
    """Return the user if credentials are valid, else ``None``."""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
