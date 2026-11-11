
"""Backend-side example using authenticated Modal SDK."""

import json
import sys

from pathlib import Path

import modal


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: py invoke_deployed.py job.json"
        )

    payload = json.loads(
        Path(sys.argv[1]).read_text(
            encoding="utf-8"
        )
    )

    fn = modal.Function.from_name(
        "opendock-boltz2-worker",
        "submit",
    )

    result = fn.remote(payload)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
