"""
Vercel serverless entry point for CausalLens API.
Wraps the FastAPI app for Vercel's Python runtime.
"""

import sys
import os

# Add project root to Python path so causallens package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.api.server import app

# Vercel looks for 'app' or 'handler'
handler = app
