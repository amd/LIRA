# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 
import argparse
import uvicorn
import os
from lira.server import openai_server


def serve():
    parser = argparse.ArgumentParser(description="Lira ASR Server")
    parser.add_argument("serve", help="Run server", nargs="?")
    parser.add_argument(
        "--backend",
        choices=["openai", "deepgram"],
        required=True,
        help="Choose which API style to expose",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type (e.g., whisper_base, zipformer)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Target device (cpu or npu)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    if args.backend == "openai":
        app = openai_server.setup_openai_server(args.model, args.device)

    elif args.backend == "deepgram":
        from lira.servers import deepgram_server

        app = deepgram_server.get_app()

    else:
        raise ValueError("Invalid backend provided")

    uvicorn.run(app, host=args.host, port=args.port)
