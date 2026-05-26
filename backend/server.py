#!/usr/bin/env python3
"""
FastAPI server entrypoint.

Usage:
    python server.py                    # Run on http://localhost:5000
    python server.py --port 8080        # Run on port 8080
    python server.py --reload           # Enable auto-reload for development
"""

import os
import argparse

import uvicorn

# Ensure CWD is the backend directory (for relative paths in config)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Orchid Classification API Server")
    parser.add_argument("--host", type=str, default=settings.api_host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting Orchid Classification API...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload or settings.api_reload}")

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload or settings.api_reload,
    )


if __name__ == "__main__":
    main()
