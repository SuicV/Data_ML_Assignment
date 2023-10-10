from fastapi import FastAPI

from src.api.routes.inference import inference_router
from src.api.constants import APP_NAME, API_PREFIX
from src.db import models
from src.db.sqlite_connector import engine

def server() -> FastAPI:
    # create db with tables
    models.Base.metadata.create_all(bind=engine)

    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    app.include_router(inference_router, prefix=API_PREFIX)
    return app