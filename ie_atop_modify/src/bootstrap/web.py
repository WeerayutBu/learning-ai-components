"""Wire the use case, model adapter, and local HTTP interface."""
from functools import partial
from pathlib import Path

from application.inference import infer_document
from infrastructure.atlop import ATLOP
from interfaces.web import create_server, parse_args


def main():
    args = parse_args()
    module = Path(__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = module / checkpoint
    model = ATLOP(module, checkpoint)
    server = create_server(args.port, partial(infer_document, predict=model))
    print(f"ATLOP demo: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
