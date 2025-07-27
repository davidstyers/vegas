#!/usr/bin/env python3
"""Test script for the Vegas CLI."""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import subprocess
import pandas as pd

# Add parent directory to path to allow importing vegas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vegas.cli.main import load_strategy_from_file, parse_date


class TestCLI(unittest.TestCase):
    """Test cases for the Vegas CLI."""
    
    def test_parse_date(self):
        """Test the parse_date function."""
        # Test valid date
        date = parse_date("2022-01-01")
        self.assertEqual(date.year, 2022)
        self.assertEqual(date.month, 1)
        self.assertEqual(date.day, 1)
        
        # Test None
        self.assertIsNone(parse_date(None))
    
    def test_load_strategy_from_file(self):
        """Test loading a strategy from a file."""
        # Create a temporary file with a strategy class
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
from vegas.strategy import Strategy, Context

class TestStrategy(Strategy):
    def initialize(self, context):
        context.symbols = ['AAPL']
        context.ma_window = 20
""")
            temp_file = f.name
        
        try:
            # Load the strategy
            strategy_class = load_strategy_from_file(temp_file)
            
            # Check that it's a Strategy subclass
            from vegas.strategy import Strategy
            self.assertTrue(issubclass(strategy_class, Strategy))
            
            # Check the class name
            self.assertEqual(strategy_class.__name__, "TestStrategy")
            
            # Create an instance and check initialization
            strategy = strategy_class()
            context = type('Context', (), {})()
            strategy.initialize(context)
            self.assertEqual(context.symbols, ['AAPL'])
            self.assertEqual(context.ma_window, 20)
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_cli_help(self):
        """Test the CLI help command."""
        # Run the CLI with --help
        result = subprocess.run(
            [sys.executable, "-m", "vegas.cli.main", "--help"],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        self.assertEqual(result.returncode, 0)
        
        # Check that the output contains expected text
        self.assertIn("Vegas Backtesting Engine CLI", result.stdout)
        self.assertIn("run", result.stdout)
    
    def test_run_help(self):
        """Test the run help command."""
        # Run the CLI with run --help
        result = subprocess.run(
            [sys.executable, "-m", "vegas.cli.main", "run", "--help"],
            capture_output=True,
            text=True
        )
        
        # Check that it ran successfully
        self.assertEqual(result.returncode, 0)
        
        # Check that the output contains expected text
        self.assertIn("strategy_file", result.stdout)
        self.assertIn("--data-file", result.stdout)
        self.assertIn("--data-dir", result.stdout)

    def test_db_status_args(self):
        """Test that db_status handles the detailed flag correctly."""
        # Import the function directly to test
        from vegas.cli.main import db_status_internal
        
        # Create a test Namespace without the detailed flag
        from argparse import Namespace
        args_without_detailed = Namespace(db_dir='db', verbose=False)
        
        # This should not raise an AttributeError
        try:
            # We don't actually run the function as it requires DB setup
            # Just check that it handles the missing attribute
            self.assertTrue(hasattr(args_without_detailed, 'detailed') == False)
            db_status_internal(args_without_detailed)
            self.assertTrue(hasattr(args_without_detailed, 'detailed'))
            self.assertEqual(args_without_detailed.detailed, False)
        except AttributeError as e:
            self.fail(f"db_status_internal raised AttributeError: {e}")
        
        # Create a test Namespace with the detailed flag
        args_with_detailed = Namespace(db_dir='db', verbose=False, detailed=True)
        
        # This should use the provided value
        try:
            self.assertEqual(args_with_detailed.detailed, True)
            self.assertTrue(getattr(args_with_detailed, 'detailed', False))
        except AttributeError as e:
            self.fail(f"Unexpected AttributeError: {e}")


if __name__ == "__main__":
    unittest.main() 