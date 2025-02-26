from abc import ABC, abstractmethod
from dataclasses import dataclass
from pandas import DataFrame
from numpy import ndarray
import numpy as np


@dataclass
class PortfolioResultsRetriever(ABC):

    portfolio_log_returns: DataFrame
    adjust_frequency: int

    @abstractmethod
    def calculate_weights(self, data: ndarray, **kwargs) -> ndarray:
        pass

    def run_backtest(self) -> ndarray:
        portfolio_returns = []
        last_weights = 0.
        for i in range(len(self.portfolio_log_returns)):
            if i % self.adjust_frequency == 0 and i > 0:
                day_weights = self.calculate_weights(
                    self.portfolio_log_returns.iloc[:i].values
                )
                last_weights = day_weights
            else:
                if i < self.adjust_frequency:
                    # Initialize weights to be equal
                    day_weights = (
                            np.ones(len(self.portfolio_log_returns.columns)) /
                            len(self.portfolio_log_returns.columns)
                    )
                else:
                    day_weights = last_weights

            day_returns = np.dot(day_weights, self.portfolio_log_returns.iloc[i])
            portfolio_returns.append(day_returns)
        return np.array(portfolio_returns)
