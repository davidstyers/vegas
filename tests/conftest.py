"""Common fixtures for tests."""

import pytest
import tempfile
import shutil
import os
from datetime import datetime
import pandas as pd

from vegas.data import DataLayer


@pytest.fixture
def test_data_layer():
    """Create a DataLayer in test mode.
    
    This ensures all database operations are performed on an in-memory database
    and temporary directory, preventing pollution of the real database.
    """
    # Create a DataLayer in test mode
    data_layer = DataLayer(test_mode=True)
    
    yield data_layer
    
    # Clean up
    data_layer.close() 