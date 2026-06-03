"""
Helm - AI-driven browser automation
===================================
"""

import os
import re
import sys
import subprocess
from dotenv import load_dotenv


def check_requirements():
    print("Checking system requirements...")

    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        return False
    print(f"  Python {sys.version.split()[0]}")

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("  Virtual environment active")
    else:
        print("  WARNING: No virtual environment detected")

    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    has_groq = bool(groq_key and groq_key != "your-groq-api-key-here")
    has_gemini = bool(gemini_key and gemini_key != "your-gemini-api-key-here")
    has_ollama = bool(ollama_url and ollama_model)

    if has_groq:
        print("  Groq API key configured")
    elif has_gemini:
        print("  Gemini API key configured")
    elif has_ollama:
        print("  Ollama configured")
    else:
        print("  WARNING: No AI provider configured")
        print("  Add GROQ_API_KEY, GEMINI_API_KEY, or OLLAMA_BASE_URL/OLLAMA_MODEL in .env")

    try:
        import fastapi
        import uvicorn
        import playwright
        import groq
        import aiosqlite
        import httpx
        from bs4 import BeautifulSoup
        print("  All packages installed")
    except ImportError as e:
        print(f"  ERROR: Missing package: {e}")
        print("  Run: pip install -r requirements.txt")
        return False

    return True


def check_playwright_browsers():
    print("Checking Playwright browsers...")
    try:
        dry_run = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "--dry-run"],
            capture_output=True, text=True, timeout=20
        )
        locations = re.findall(r"Install location:\s*(.+)", dry_run.stdout)
        chromium_locations = [
            path.strip() for path in locations
            if "chromium" in path.lower()
        ]
        if chromium_locations and any(os.path.exists(path) for path in chromium_locations):
            print("  Playwright browsers ready")
            return True

        print("  Installing Playwright Chromium...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print("  ERROR: Failed to install Chromium")
            details = (result.stderr or result.stdout).strip()
            if details:
                print(f"  {details.splitlines()[-1]}")
            return False
        print("  Playwright browsers ready")
    except subprocess.TimeoutExpired:
        print("  ERROR: Playwright browser setup timed out")
        print("  Try running: python -m playwright install chromium")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    return True


def display_startup_info(port: int):
    print()
    print("=" * 60)
    print("  HELM v0.2 - browse the web with AI")
    print("=" * 60)
    print()
    print("  Features:")
    print("    - Real AI-powered browser automation")
    print("    - Live browser preview & screenshots")
    print("    - Task history & analytics dashboard")
    print("    - Session recording & export (Python/JSON)")
    print("    - Task templates & workflow builder")
    print("    - Scheduled automations")
    print("    - Smart data extraction (CSV/JSON/Markdown)")
    print("    - Voice input & dark mode")
    print("    - Self-healing selectors & error recovery")
    print("    - Multi-provider AI (Groq/Gemini/Ollama)")
    print()
    print(f"  Web interface:  http://localhost:{port}")
    print(f"  WebSocket API:  ws://localhost:{port}/ws/advanced")
    print(f"  REST API:       http://localhost:{port}/api/")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()


def main():
    print("Helm starting...")
    print()

    if not check_requirements():
        print("\nRequirements check failed.")
        return 1

    if not check_playwright_browsers():
        print("\nPlaywright setup failed. Run: playwright install")
        return 1

    port = int(os.environ.get("PORT", 8000))
    display_startup_info(port)

    try:
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
    except Exception as e:
        print(f"\nFailed to start: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
