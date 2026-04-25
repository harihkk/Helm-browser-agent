"""
Surf - AI-driven browser automation
===================================
"""

import os
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

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your-groq-api-key-here":
        print("  ERROR: Groq API key not found")
        print("  Set GROQ_API_KEY in .env (get it from https://console.groq.com/keys)")
        return False
    print("  Groq API key configured")

    try:
        import fastapi
        import uvicorn
        import playwright
        import groq
        import aiosqlite
        import httpx
        from pydantic_settings import BaseSettings
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
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run"],
            capture_output=True, text=True
        )
        if "chromium" not in result.stdout.lower():
            print("  Installing Playwright browsers...")
            install_result = subprocess.run(
                [sys.executable, "-m", "playwright", "install"],
                capture_output=True, text=True
            )
            if install_result.returncode != 0:
                print("  ERROR: Failed to install browsers")
                return False
        print("  Playwright browsers ready")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    return True


def display_startup_info():
    print()
    print("=" * 60)
    print("  SURF v0.2 - browse the web with AI")
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
    print("    - Multi-provider AI (Groq/Ollama/Gemini)")
    print()
    print("  Web interface:  http://localhost:8000")
    print("  WebSocket API:  ws://localhost:8000/ws/advanced")
    print("  REST API:       http://localhost:8000/api/")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()


def main():
    print("Surf starting...")
    print()

    if not check_requirements():
        print("\nRequirements check failed.")
        return 1

    if not check_playwright_browsers():
        print("\nPlaywright setup failed. Run: playwright install")
        return 1

    display_startup_info()

    try:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
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
