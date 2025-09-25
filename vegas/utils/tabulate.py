from tabulate import tabulate

def tabulate_results(results: dict) -> str:
    """Tabulate the results of a backtest."""
    table_data = [(k, round(v, 3)) for k, v in results.items()]
    
    return tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid")
