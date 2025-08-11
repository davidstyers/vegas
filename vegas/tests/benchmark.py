import pytest
import polars as pl
import time
from datetime import date
from vegas.pipeline.engine import PipelineEngine
from vegas.pipeline.pipeline import Pipeline
from vegas.pipeline.factors.basic import SimpleMovingAverage
from vegas.data.data_layer import DataLayer

# Define a mock data layer for benchmarking
class MockDataLayer(DataLayer):
    def __init__(self, num_rows, num_symbols):
        super().__init__()
        self.num_rows = num_rows
        self.num_symbols = num_symbols
        self.df = self._generate_data()

    def _generate_data(self):
        # Generate a large dataset for benchmarking
        symbols = [f'SYM{i}' for i in range(self.num_symbols)]
        dates = pl.date_range(date(2000, 1, 1), date(2023, 12, 31), "1d", eager=True)
        
        data = {
            'date': dates.to_list() * self.num_symbols,
            'symbol': [s for s in symbols for _ in range(len(dates))],
            'close': [100 + i for i in range(len(dates) * self.num_symbols)],
            'volume': [1000 + i for i in range(len(dates) * self.num_symbols)]
        }
        
        return pl.DataFrame(data)

    def get_data_for_backtest(self, start_date, end_date):
        if start_date is None or end_date is None:
            return self.df
        return self.df.filter((pl.col('date') >= start_date) & (pl.col('date') <= end_date))

# Benchmark the new Polars-native pipeline engine
def benchmark_new_pipeline(num_rows=10000, num_symbols=100):
    
    # Create a mock data layer with a large dataset
    data_layer = MockDataLayer(num_rows, num_symbols)
    
    # Create a pipeline with a 50-day SMA factor
    pipeline = Pipeline(columns={
        'sma_50': SimpleMovingAverage(window_length=50)
    })
    
    # Create a new pipeline engine
    engine = PipelineEngine(data_layer)
    
    # Run the pipeline and measure the execution time
    start_time = time.time()
    engine.run_pipeline(pipeline, date(2000, 1, 1), date(2023, 12, 31))
    end_time = time.time()
    
    return end_time - start_time

if __name__ == "__main__":
    # Run the benchmark for the new pipeline engine
    new_time = benchmark_new_pipeline()
    print(f"New pipeline engine execution time: {new_time:.4f} seconds")