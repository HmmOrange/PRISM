"""Standalone auth server exposing /auth/* endpoints."""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from utils.constants import SETTINGS
from api.auth import router as auth_router


def get_application():
    app = FastAPI(
        title="PRISM Auth Server",
        description="Authentication Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.CORS_ALLOW_ORIGINS,
        allow_methods=SETTINGS.CORS_ALLOW_METHODS,
        allow_headers=SETTINGS.CORS_ALLOW_HEADERS,
        allow_credentials=True,
    )

    app.include_router(auth_router)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "auth"}

    return app


app = get_application()
