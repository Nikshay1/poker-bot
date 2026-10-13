"""A small local server behind the advisor page.

Standard library only. The page sends the hand as entered so far and gets
back what the hand is waiting for, with a recommendation when it is the
hero's turn. The server keeps no hand state, so undo is just a shorter list.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from pokerbot.advisor import SolverConfig
from pokerbot.cards import parse_cards
from pokerbot.game.state import SpotError
from pokerbot.live import LiveHand, advise_live, hand_from_payload
from pokerbot.policy.multiway import advise_multiway

MAX_BODY = 64 * 1024
# The Numba kernels are not safe to run from several threads at once.
_compute = threading.Lock()


@lru_cache(maxsize=512)
def _hand_reply(hand: LiveHand, time_budget: float) -> dict:
    return advise_live(hand, SolverConfig(time_budget=time_budget)).to_dict()


def hand_reply(data: dict, time_budget: float) -> dict:
    return _hand_reply(hand_from_payload(data), time_budget)


def multiway_reply(data: dict) -> dict:
    try:
        hole = parse_cards(str(data["hole"]))
        board = parse_cards(str(data.get("board", "")))
        players = int(data.get("players", 3))
        pot, to_call = int(data["pot"]), int(data.get("to_call") or 0)
        stack = int(data["stack"]) if data.get("stack") else None
        big_blind = int(data.get("big_blind", 100))
    except (KeyError, TypeError, ValueError) as err:
        raise SpotError("bad_input", f"could not read the situation: {err}") from err
    if len(hole) != 2:
        raise SpotError("bad_hole", "give exactly two hole cards")
    if len(board) not in (0, 3, 4, 5):
        raise SpotError("bad_board", "the board must have 0, 3, 4 or 5 cards")
    rec = advise_multiway(
        hole, board, pot=pot, to_call=to_call, stack=stack, n_opponents=players - 1,
        big_blind=big_blind,
    )  # fmt: skip
    return {**rec.to_dict(), "text": str(rec.action)}


def make_handler(time_budget: float):
    page = resources.files("pokerbot.web").joinpath("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "not_found", "message": "no such page"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    raise SpotError("bad_input", "the request body is missing or too large")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise SpotError("bad_input", "the request must be a JSON object")
                with _compute:
                    if self.path == "/api/hand":
                        reply = hand_reply(data, time_budget)
                    elif self.path == "/api/multiway":
                        reply = multiway_reply(data)
                    else:
                        self._json(404, {"error": "not_found", "message": "no such endpoint"})
                        return
            except SpotError as err:
                self._json(400, err.as_dict())
            except json.JSONDecodeError as err:
                self._json(400, {"error": "bad_input", "message": f"not valid JSON: {err}"})
            else:
                self._json(200, reply)

        def log_message(self, format, *args):
            """Stay quiet; one line per click would bury the address."""

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, time_budget: float = 2.0, browser=True):
    server = ThreadingHTTPServer((host, port), make_handler(time_budget))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{server.server_address[1]}/"
    print(f"pokerbot advisor at {url}  (Ctrl+C to stop)")
    if host == "0.0.0.0":
        print("Listening on all interfaces: open http://<this machine's address>:"
              f"{server.server_address[1]}/ from a phone on the same network.")  # fmt: skip
    if browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
