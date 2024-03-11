#!/usr/bin/env python

from baby_degen.strategies.sma_strategy import sma_strategy
from dotenv import load_dotenv
from utils import get_logger

load_dotenv()
logger = get_logger(__name__)


def strategy_map(strategy):
    """Map the strategy name to the strategy class."""

    strategy_dict = {
        "sma-strategy": sma_strategy,
    }

    strategy = strategy_dict.get(strategy, None) 

    if strategy is None:
        raise Exception(f"Strategy {strategy} not found.")
    else:
        return strategy

def run_benchmark(kwargs):
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""

    logger.info("Running benchmark tests...")

    strategies = kwargs.pop("strategies")

    for t in strategies:
        strategy = strategy_map(t)
        response = strategy.run(**kwargs)
        print('=======', response)

if __name__ == "__main__":
    kwargs = {}
    kwargs["strategies"] = [
        "sma-strategy",
    ]
    kwargs["transformed_data"] = "baby_degen/strategies/data/olas_5m.json"
    kwargs["token_id"] = "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"
    kwargs["portfolio_data"] = {
        "So11111111111111111111111111111111111111112": 10,
        "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": 0,
    }
    run_benchmark(kwargs)