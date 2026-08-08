"""Run the RiskLens UI prototype locally.

This intentionally serves only static files. The interface uses mock data so
UI/UX can be tested independently of model-development work.

Run: python app.py
Open: http://localhost:8000/riskLens.html
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

os.chdir(Path(__file__).parent / "static")

if __name__ == "__main__":
    server = ThreadingHTTPServer(("", 8000), SimpleHTTPRequestHandler)
    print("RiskLens UI prototype: http://localhost:8000/riskLens.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
