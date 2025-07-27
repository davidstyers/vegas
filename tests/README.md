# Vegas Testing Framework

This directory contains test code for the Vegas backtesting engine.

## Best Practices

### Database Testing Isolation

Tests in Vegas use isolated test databases to prevent polluting the production database. This is accomplished through several approaches:

1. **Test Mode in DataLayer**: 
   ```python
   # Create a DataLayer in test mode
   data_layer = DataLayer(test_mode=True)
   ```
   
   When test_mode=True:
   - Uses an in-memory DuckDB database
   - Creates temporary directories for Parquet files
   - Automatically cleans up resources after tests

2. **Fixture for Test DataLayer**:  
   Use the `test_data_layer` fixture from `tests/conftest.py`:
   ```python
   def test_something(test_data_layer):
       # test_data_layer is already configured for isolation
       test_data_layer.load_data(...)
   ```

3. **DatabaseManager Isolation**: 
   For direct database tests, use the helper function in `tests/test_database_duplicates.py`:
   ```python
   # Create an isolated test database manager
   db = create_test_db_manager(temp_dir)
   ```

### General Guidelines

1. Always use isolated test databases - never write tests that modify the real database.
2. Use temporary directories for file operations.
3. Clean up resources after tests complete.
4. Each test should be independent and not rely on state from other tests.

## Running Tests

Run all tests:
```bash
python -m pytest
```

Run specific test file:
```bash
python -m pytest tests/test_data_layer.py
```

Run specific test:
```bash
python -m pytest tests/test_data_layer.py::test_handling_timezone_information
```

With verbose output:
```bash
python -m pytest -v
``` 