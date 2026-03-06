import contextlib
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

async def root(_request):
    return JSONResponse({"status": "ok", "name": "ares-test"})

async def mcp_probe(_request):
    return PlainTextResponse("MCP ROUTE EXISTS")

app = Starlette(
    routes=[
        Route("/", root),
        Route("/mcp", mcp_probe, methods=["GET", "POST"]),
    ]
)
