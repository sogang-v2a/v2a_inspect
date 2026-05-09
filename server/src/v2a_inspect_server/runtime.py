from __future__ import annotations

import argparse
import uvicorn

from v2a_inspect_server.settings import settings

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v2a-inspect-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the server runtime HTTP API")
    serve_parser.add_argument("--host", type=str, default=settings.host, help="Bind host")
    serve_parser.add_argument("--port", type=int, default=settings.port, help="Bind port")

    args = parser.parse_args(argv)

    if args.command == "serve":
        uvicorn.run("v2a_inspect_server.app:app", host=args.host, port=args.port, reload=False)
        return 0
    return 1

if __name__ == "__main__":
    main()
