"""Run the evaluation matches and save the results as JSON.

Every match is played as duplicate pairs. The seed given here must not have
been used for tuning: the numbers reported in docs/RESULTS.md come from one
run of this script on a fresh seed, after the bots were frozen.

    uv run python scripts/evaluate.py --seed 1000 --workers 16
"""

import argparse
import json
import pathlib
import subprocess
import time

from pokerbot.arena.match import run_match

ROOT = pathlib.Path(__file__).resolve().parents[1]
# (bot A, bot B, duplicate pairs). Matches between very unequal bots need
# fewer hands because the difference is large.
MATCHES = [
    ("solver", "heuristic", 10000),
    ("solver", "sharp", 5000),
    ("solver", "equity", 3000),
    ("solver", "call", 1500),
    ("solver", "random", 1500),
    ("heuristic", "equity", 3000),
    ("heuristic", "call", 1500),
    ("heuristic", "random", 1500),
    ("equity", "call", 1500),
    ("equity", "random", 1500),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every pair count")
    parser.add_argument("--only", help="run only matches whose first bot has this name")
    args = parser.parse_args()

    out = ROOT / f"docs/data/matches_seed{args.seed}.json"
    results = json.loads(out.read_text())["matches"] if out.exists() else []
    done = {(r["bot_a"], r["bot_b"]) for r in results}

    def git(*command: str) -> str:
        result = subprocess.run(
            ["git", *command], capture_output=True, text=True, cwd=ROOT, check=True
        )
        return result.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "src"))
    for bot_a, bot_b, pairs in MATCHES:
        if (bot_a, bot_b) in done or (args.only and bot_a != args.only):
            continue
        result = run_match(bot_a, bot_b, int(pairs * args.scale), args.seed, workers=args.workers)
        print(result.summary(), flush=True)
        results.append(result.to_dict())
        # Saved after every match, so an interrupted run can be resumed.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "commit": commit,
                    "uncommitted_source_changes": dirty,
                    "finished": time.strftime("%Y-%m-%d %H:%M"),
                    "matches": results,
                },
                indent=1,
            )
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
