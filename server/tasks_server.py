from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from utils.constants import SETTINGS
from server.schema_base import ResponseSchemaBase
from server.api_router import router


class CustomException(Exception):
    http_code: int
    code: str
    message: str

    def __init__(self, http_code: int = 0, code: str = "", message: str = ""):
        self.http_code = http_code if http_code else 500
        self.code = code if code else str(self.http_code)
        self.message = message


async def http_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, CustomException):
        return JSONResponse(
            status_code=exc.http_code,
            content=jsonable_encoder(
                ResponseSchemaBase().custom_response(exc.code, exc.message)
            ),
        )

    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(
            ResponseSchemaBase().custom_response("500", "Internal Server Error")
        ),
    )

def get_application():
    app = FastAPI(
        title="PRISM Tasks Server",
        description="Tasks Server",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.CORS_ALLOW_ORIGINS,
        allow_methods=SETTINGS.CORS_ALLOW_METHODS,
        allow_headers=SETTINGS.CORS_ALLOW_HEADERS,
        allow_credentials=SETTINGS.CORS_ALLOW_CREDENTIALS,
    )

    app.include_router(router)
    return app


app = get_application()
