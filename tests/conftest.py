"""Pytest configuration for FinShield tests."""
import os

# Set test mode to use in-memory DuckDB
os.environ["FINSHIELD_TEST_MODE"] = "1"