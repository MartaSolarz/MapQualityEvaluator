"""Entry point: python3 -m viewer [--port 8060]"""

import argparse
import sys
import webbrowser
from pathlib import Path
from http.server import HTTPServer

sys.path.insert(0, str(Path(__file__).parent.parent))

from .server import ViewerHandler


def main():
    import db
    db.init_db()

    parser = argparse.ArgumentParser(description="Stage 6 viewer")
    parser.add_argument("--port", type=int, default=8060)
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
