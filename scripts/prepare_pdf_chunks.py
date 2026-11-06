#!/usr/bin/env python3
"""Run PDF ingestion from any working directory."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from services.ingestion.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
