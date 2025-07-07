#!/usr/bin/env python3
"""Main entry point for the Vegas backtesting engine.

This module forwards to the CLI module for command-line usage.
"""

import sys
from vegas.cli.main import main

if __name__ == "__main__":
    sys.exit(main()) 