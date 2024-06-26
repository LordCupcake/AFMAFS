from dataclasses import dataclass, field
from datetime import date
from typing import Optional

@dataclass
class StockPrediction:
    """
    Represents a stock prediction project with various parameters.
    """
    ticker: str
    start_date: date
    validation_date: Optional[date] = None
    project_folder: str = ""
    github_url: Optional[str] = None
    epochs: int = 10
    time_steps: int = 1
    token: Optional[str] = None
    batch_size: int = 32

    def set_ticker(self, value: str):
        """
        Set the stock ticker symbol.
        """
        if not isinstance(value, str):
            raise ValueError("Ticker must be a string.")
        self.ticker = value.upper()  # Ensure the ticker is uppercase

    def set_start_date(self, value: date):
        """
        Set the start date for data collection.
        """
        if not isinstance(value, date):
            raise ValueError("Start date must be a valid date object.")
        self.start_date = value

    def set_validation_date(self, value: date):
        """
        Set the validation date for the model.
        """
        if not isinstance(value, date):
            raise ValueError("Validation date must be a valid date object.")
        self.validation_date = value
