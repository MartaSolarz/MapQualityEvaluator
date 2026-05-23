"""Entry point: python3 -m viewer [--port 8050]"""

import argparse
import webbrowser
from http.server import HTTPServer

from .data import init_dataset_table
from .builder import init_builder_tables
from .server import ViewerHandler


def main():
    init_dataset_table()
    init_builder_tables()

    parser = argparse.ArgumentParser(description="Stage 5 results viewer")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"Opening viewer at {url}")
    print("Press Ctrl+C to stop.\n")

    webbrowser.open(url)
    server = HTTPServer(("localhost", args.port), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
