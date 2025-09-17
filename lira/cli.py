# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from lira.models.whisper.transcribe import WhisperONNX
from lira.models.zipformer.transcribe import ZipformerONNX
from lira.server.server import serve


def main():
    parser = argparse.ArgumentParser(prog="lira", description="LIRA CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Top-level commands")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run a single model + chain tools")
    run_subparsers = run_parser.add_subparsers(dest="model", help="Model to run")
    WhisperONNX.parse_cli(run_subparsers)
    ZipformerONNX.parse_cli(run_subparsers)

    # 'compare' command
    compare_parser = subparsers.add_parser(
        "compare", help="Run multiple models simultaneously + chain tools"
    )
    compare_parser.add_argument(
        "--models", nargs="+", required=True, help="List of models to compare"
    )
    compare_parser.add_argument(
        "--input", required=True, help="Path to input audio file"
    )

    # 'tool' command
    tool_parser = subparsers.add_parser(
        "tool", help="Run standalone tools or chain them"
    )
    tool_parser.add_argument("--name", required=True, help="Name of the tool to run")
    tool_parser.add_argument(
        "--args", nargs=argparse.REMAINDER, help="Arguments for the tool"
    )

    # 'serve' command
    serve_parser = subparsers.add_parser("serve", help="Run the LIRA server")
    serve_parser.add_argument(
        "--backend", required=True, help="Backend to use (e.g., openai)"
    )
    serve_parser.add_argument(
        "--model", required=True, help="Model type (e.g., whisper-base)"
    )
    serve_parser.add_argument(
        "--device", default="cpu", help="Device to run the server on"
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to"
    )
    serve_parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )

    args = parser.parse_args()

    if args.command == "run":
        if hasattr(args, "func"):
            args.func(args)
    elif args.command == "compare":
        compare_models(args)
    elif args.command == "tool":
        run_tool(args)
    elif args.command == "serve":
        serve()
    else:
        parser.print_help()


def compare_models(args):
    print(f"Comparing models {args.models} on {args.input}")


def run_tool(args):
    print(f"Running tool {args.name} with arguments {args.args}")


if __name__ == "__main__":
    main()
