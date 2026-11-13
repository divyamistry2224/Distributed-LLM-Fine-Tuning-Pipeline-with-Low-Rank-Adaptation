"""Adapter-backed FastAPI service with bounded input and optional bearer auth."""

from contextlib import asynccontextmanager
import hmac
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


class GenerateRequest(BaseModel):
    context: str = Field(min_length=1, max_length=100000)
    question: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)


class GenerateResponse(BaseModel):
    answer: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float


def create_app(engine=None, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            from src.inference import GenerationEngine

            base, adapter = os.getenv("LORA_BASE_MODEL"), os.getenv("LORA_ADAPTER_DIR")
            if not base or not adapter:
                raise RuntimeError("Set LORA_BASE_MODEL and LORA_ADAPTER_DIR before starting the service")
            app.state.engine = await run_in_threadpool(GenerationEngine, base, adapter)
        yield

    app = FastAPI(title="Document LoRA Inference", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.api_key = api_key if api_key is not None else os.getenv("LORA_API_KEY")

    @app.get("/health")
    def health():
        return {
            "status": "ready" if app.state.engine is not None else "starting",
            "adapter_loaded": app.state.engine is not None,
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest, authorization: str | None = Header(default=None)):
        key = app.state.api_key
        if key and not hmac.compare_digest(authorization or "", "Bearer " + key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            return await run_in_threadpool(
                app.state.engine.generate, request.context, request.question, request.max_new_tokens
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
