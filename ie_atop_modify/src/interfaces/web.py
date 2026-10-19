"""One local page and one inference endpoint; no model dependencies."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--checkpoint", default="checkpoints/spans-bert.pt")
    return parser.parse_args()


def create_server(port, infer):
    page = Path(__file__).with_suffix('.html').read_bytes()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/relations":
                self.send(200, json.loads((Path(__file__).resolve().parents[2] / "data/relations.json").read_text()))
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/infer":
                self.send(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send(403, {"error": "Use this page to run inference."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(415, {"error": "Send JSON."})
                return
            if not lock.acquire(blocking=False):
                self.send(409, {"error": "Another inference is running. Try again shortly."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 64000:
                    raise ValueError("Request is empty or too large.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object with text and spans.")
                result = infer(payload)
                self.send(200, result)
            except (ValueError, UnicodeDecodeError) as error:
                self.send(400, {"error": str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.send(500, {"error": "Inference failed. See the server log."})
            finally:
                lock.release()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
