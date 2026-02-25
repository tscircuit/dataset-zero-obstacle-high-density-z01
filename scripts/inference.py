#!/usr/bin/env python3
"""
Local inference script for the PCB routing model.

Sends a connection-pairs PNG to the deployed Modal endpoint and saves the
routed result.

Usage:
    python scripts/inference.py input.png output.png [--strength 0.75] [--seed 42]

Requires the Modal endpoint to be deployed first:
    modal deploy scripts/deploy_api.py
"""

import argparse
import base64
import json
import sys

import requests

# Update this after deploying with `modal deploy scripts/deploy_api.py`
ENDPOINT_URL = "https://YOUR_MODAL_USERNAME--pcbrouter-flux2-inference-route.modal.run/"

DEFAULT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)


def main():
    parser = argparse.ArgumentParser(
        description="Route PCB traces using the fine-tuned FLUX.2 Klein model"
    )
    parser.add_argument("input", help="Path to the connection-pairs PNG image")
    parser.add_argument("output", help="Path to save the routed PNG image")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="img2img strength (0.0=no change, 1.0=full generation). Default: 0.75",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Edit instruction",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=ENDPOINT_URL,
        help="Modal endpoint URL",
    )
    args = parser.parse_args()

    # Read and encode input image
    with open(args.input, "rb") as f:
        input_b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "input_image": input_b64,
        "instruction": args.instruction,
        "strength": args.strength,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    print(f"Sending {args.input} to endpoint...")
    print(f"  strength={args.strength}, seed={args.seed}")

    # Stream SSE response
    resp = requests.post(
        args.endpoint,
        json=payload,
        headers={"Accept": "text/event-stream"},
        stream=True,
    )
    resp.raise_for_status()

    result_image = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[6:])

        if event["stage"] == "generating":
            pct = round(event["progress"] * 100)
            print(f"  Step {event['step']}/{event['total']} ({pct}%)", end="\r")
        elif event["stage"] == "complete":
            print(f"\n  Done!")
            result_image = base64.b64decode(event["image"])
        elif event["stage"] == "error":
            print(f"\n  Error: {event['message']}", file=sys.stderr)
            sys.exit(1)

    if result_image:
        with open(args.output, "wb") as f:
            f.write(result_image)
        print(f"Saved routed image to {args.output}")
    else:
        print("No image received from endpoint", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
