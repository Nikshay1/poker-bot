"""Print the match table of an evaluation run as Markdown.

    uv run python scripts/report.py docs/data/matches_seed1000.json
"""

import json
import pathlib
import sys


def main():
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
    print(f"Seed {data['seed']}, commit {data['commit']}, finished {data['finished']}.\n")
    print("| Bot | Opponent | Win rate, bb/100 | 95% interval | Hands | Verdict |")
    print("|---|---|---|---|---|---|")
    for m in data["matches"]:
        low, high = m["bb_per_100"] - m["ci95"], m["bb_per_100"] + m["ci95"]
        verdict = "wins" if low > 0 else "loses" if high < 0 else "not separated"
        print(
            f"| {m['bot_a']} | {m['bot_b']} | {m['bb_per_100']:+.1f} | "
            f"{low:+.1f} to {high:+.1f} | {m['hands']:,} | {verdict} |"
        )
    illegal = sum(m["illegal_a"] + m["illegal_b"] for m in data["matches"])
    print(f"\nIllegal actions across all matches: {illegal}.")


if __name__ == "__main__":
    main()
