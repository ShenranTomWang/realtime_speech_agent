"""Authenticated model-only ingress. No agent harness executes on this server."""

import asyncio
import contextlib
import hmac
import os

import httpx
import websockets
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

app = FastAPI()


def authorized(value):
    return hmac.compare_digest(value or "", "Bearer " + os.environ["SOULX_MODEL_TOKEN"])


@app.middleware("http")
async def authenticate(request, call_next):
    if not authorized(request.headers.get("authorization")):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/configuration")
async def configuration():
    from pathlib import Path

    return {
        "frontend_model": os.environ["SOULX_FRONTEND_MODEL"],
        "frontend_tools": yaml.safe_load(Path(os.environ["SOULX_TOOLS_CONFIG"]).read_text()),
    }


async def forward(request, url):
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=20), trust_env=False)
    try:
        upstream = await client.send(
            client.build_request(
                request.method, url, content=await request.body(),
                headers={"Content-Type": request.headers.get("content-type", "application/json")},
                params=request.query_params,
            ),
            stream=True,
        )
    except Exception:
        await client.aclose()
        raise

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body(), status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.api_route("/llm/v1/{path:path}", methods=["GET", "POST"])
async def llm(path: str, request: Request):
    if path not in {"chat/completions", "models"}:
        raise HTTPException(404)
    return await forward(request, os.environ["SOULX_FRONTEND_BASE_URL"] + "/" + path)


@app.post("/tts")
async def tts(request: Request):
    return await forward(request, os.environ["SOULX_TTS_URL"])


@app.websocket("/turn")
async def turn(ws: WebSocket):
    if not authorized(ws.headers.get("authorization")):
        await ws.close(code=1008)
        return
    await ws.accept()
    tasks = []
    try:
        async with websockets.connect(os.environ["SOULX_VAD_URL"]) as upstream:
            async def upload():
                while True:
                    await upstream.send(await ws.receive_text())

            async def download():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await ws.send_bytes(message)
                    else:
                        await ws.send_text(message)

            tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(RuntimeError):
            await ws.close()
