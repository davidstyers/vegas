class MyPipeline:
    def __init__(self, data_portal, frequency='1d'):
        self.assets = ["AAPL", "GOOG"]
        self.fields = ["price"]
        self.data_portal = data_portal
        self.frequency = frequency
        # bar_count is needed for history call, based on the prompt's signature
        self.bar_count = 3 

    def run(self):
        # The call to the data portal is now flexible
        data = self.data_portal.history(
            assets=self.assets,
            fields=self.fields,
            bar_count=self.bar_count,
            frequency=self.frequency
        )
        print(f"Processing {self.frequency} data: {data}")
        # ... further processing logic