"""Add project root to sys.path; mock external API deps before any import."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Must be before any project import so modules load with mocked dependencies
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub heavy optional dependencies that may not be installed in test env
for _mod in ("openai", "ccxt", "ta", "duckduckgo_search", "gspread", "streamlit"):
    sys.modules.setdefault(_mod, MagicMock())
