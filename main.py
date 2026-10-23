"""Run the four-stage workflow or a mixed workload over one SIE server."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

from inference.client import Models, SIEClient
from inference.replay import ReplayServer
from inference.workflow import load_case, mixed_request, run_pipeline
from inference.workload import run_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "benchmark"], nargs="?", default="demo")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=16,
                        help="benchmark inference calls, or demo workflows (four calls each)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--loading-retries", type=int, default=60)
    parser.add_argument("--output", type=Path, help="optional JSON metrics file")
    for field, model in vars(Models()).items():
        parser.add_argument("--" + field, default=model)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    models = Models(**{field: getattr(args, field) for field in vars(Models())})
    if args.mode == "replay" and models != Models():
        parser.error("replay fixtures support the default four models only")
    case = load_case()
    context = ReplayServer() if args.mode == "replay" else nullcontext(None)
    with context as server:
        url = server.url if server else args.url
        client = SIEClient(url, api_key=os.environ.get("SIE_API_KEY") if server is None else None,
                           timeout=args.timeout, loading_retries=args.loading_retries)
        if not client.ready():
            parser.exit(2, "SIE readiness check failed; start the server and verify /readyz.\n")
        if args.command == "demo":
            operation = lambda _index: run_pipeline(client, case, models)
        else:
            operation = lambda index: mixed_request(client, case, index, models)
        report = run_workload(operation, args.requests, args.concurrency)
        report.update(mode=args.mode, workload=args.command, models=vars(models), http=client.stats(),
                      measurement=("Synthetic localhost HTTP fixture timings; no model or GPU inference"
                                   if server else "Live SIE client measurements; client wall-clock timings"))
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
