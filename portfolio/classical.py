import pandas as pd

from .base import PortfolioResultsRetriever
import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from pandas import DataFrame


class MarkovitzPortfolio(PortfolioResultsRetriever):
    """
    Markovitz Portfolio class that calculates the weights for the portfolio using the Markovitz Portfolio theory.
    """

    def calculate_weights(self, data: ndarray, **kwargs) -> ndarray:
        cov = np.cov(data, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(cov.shape[0])
        # avoid short selling
        weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
        corrected_weights = np.maximum(weights, 0)
        return weights / np.sum(corrected_weights)


class EqualWeightsPortfolio(PortfolioResultsRetriever):
    """
    Equal weights portfolio class that calculates the weights for the portfolio using equal weights.
    """

    def calculate_weights(self, data: ndarray, **kwargs) -> ndarray:
        return np.ones(data.shape[1]) / data.shape[1]


class RiskParityPortfolio(PortfolioResultsRetriever):
    """
    Risk parity portfolio class that calculates the weights for the portfolio using risk parity.
    """

    def calculate_weights(self, data: ndarray, **kwargs) -> ndarray:
        cov = np.cov(data, rowvar=False)
        vol = np.sqrt(np.diag(cov))
        inv_vol = 1 / (vol + 1e-12)
        return inv_vol / np.sum(inv_vol)


@dataclass
class CAPMPortfolio(PortfolioResultsRetriever):
    """
    CAPM Portfolio class that calculates the weights for the portfolio using the CAPM model.
    """

    risk_free_rate = 0.02
    market_return = 0.08

    def calculate_weights(self, data: ndarray, **kwargs) -> ndarray:
        tickers = self.portfolio_log_returns.columns
        historical_market_returns = data.mean(axis=1)

        # Step 2: Estimate Beta for each stock relative to the market
        betas = {}
        for j in range(data.shape[1]):
            cov = np.cov(data[:, j], historical_market_returns)
            beta = cov[0, 1] / np.var(historical_market_returns)
            betas[tickers[j]] = beta

        # Step 3: Compute Expected Returns Using CAPM
        expected_returns = {}
        for ticker in tickers:
            expected_return = self.risk_free_rate + betas[ticker] * (self.market_return - self.risk_free_rate)
            expected_returns[ticker] = expected_return
        expected_returns_array = np.array(list(expected_returns.values()))
        return expected_returns_array / np.sum(expected_returns_array)
