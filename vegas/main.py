#!/usr/bin/env python3
"""Main entry point for running the Vegas CLI directly."""

import sys

def main():
    """Execute the Vegas CLI main function."""
    from vegas.cli.main import main as cli_main
    return cli_main()

if __name__ == "__main__":
    from vegas.cli.main import main
    main() 