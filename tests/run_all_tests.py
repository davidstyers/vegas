#!/usr/bin/env python3
"""Main test runner for the Vegas backtesting engine test suite."""

import os
import sys
import pytest
import argparse


def run_tests(test_type=None, verbose=False, coverage=False):
    """Run the Vegas test suite.
    
    Args:
        test_type: Type of tests to run ('core', 'integration', 'specialized', or None for all)
        verbose: Whether to display verbose output
        coverage: Whether to generate coverage report
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Build pytest arguments
    pytest_args = []
    
    # Set test path based on type
    if test_type == 'core':
        pytest_args.extend([
            "tests/test_database.py",
            "tests/test_data_layer.py",
            "tests/test_engine.py",
            "tests/test_portfolio.py",
            "tests/test_broker.py",
            "tests/test_cli.py"
        ])
    elif test_type == 'integration':
        pytest_args.extend([
            "tests/test_integration.py",
            "tests/test_database_extended.py"
        ])
    elif test_type == 'specialized':
        pytest_args.extend([
            "tests/test_bias_prevention.py",
            "tests/test_performance.py",
            "tests/test_robustness.py"
        ])
    else:
        # Run all tests in the tests directory
        pytest_args.append("tests")
    
    # Add verbosity
    if verbose:
        pytest_args.append("-v")
    
    # Add coverage if requested
    if coverage:
        pytest_args.extend(["--cov=vegas", "--cov-report=term", "--cov-report=html"])
    
    # Run pytest
    print(f"Running Vegas test suite with args: {' '.join(pytest_args)}")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vegas backtesting engine test suite")
    parser.add_argument(
        "--type", 
        choices=["core", "integration", "specialized", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Display verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Run a specific test file"
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Run a specific test file
        test_path = f"tests/{args.file}" if not args.file.startswith("tests/") else args.file
        pytest_args = [test_path]
        
        if args.verbose:
            pytest_args.append("-v")
            
        if args.coverage:
            pytest_args.extend(["--cov=vegas", "--cov-report=term", "--cov-report=html"])
            
        print(f"Running specific test file: {' '.join(pytest_args)}")
        sys.exit(pytest.main(pytest_args))
    else:
        # Convert 'all' to None for run_tests function
        test_type = args.type if args.type != "all" else None
        
        # Run tests and exit with the appropriate code
        sys.exit(run_tests(test_type, args.verbose, args.coverage)) 