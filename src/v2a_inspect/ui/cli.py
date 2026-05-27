from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2a-inspect UI server.")
    parser.add_argument("--host", default=os.getenv("V2A_INSPECT_UI_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        default=int(os.getenv("V2A_INSPECT_UI_PORT", "8501")),
        type=int,
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "v2a_inspect.ui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
