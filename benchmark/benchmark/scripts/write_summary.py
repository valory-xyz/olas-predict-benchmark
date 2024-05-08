#!/usr/bin/env python

from benchmark.utils import get_logger, TokenCounterCallback
import os
import pandas as pd
from pathlib import Path

logger = get_logger(__name__)

def write_results(csv_file_path):

    results_path = Path(csv_file_path.parent)
    time_string = csv_file_path.stem.split("_", 1)[-1]

    results_df = pd.read_csv(csv_file_path)
    num_errors = results_df["error"].count()
    logger.info(f"Num errors: {str(num_errors)}")
    results_df = results_df.dropna(subset=["prediction"])
    grouped_df = results_df.groupby(["tool", "model"]).agg(
        {
            "Correct": ["mean", "sum", "count"],
            "crowd_correct": ["mean"],
            "input_tokens": ["mean"],
            "output_tokens": ["mean"],
            "total_tokens": ["mean"],
            "input_cost": ["mean"],
            "output_cost": ["mean"],
            "total_cost": ["mean"],
        }
    )

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]
    summary_df = grouped_df.reset_index().rename(
        columns={
            "Correct_mean": "accuracy",
            "Correct_sum": "correct",
            "Correct_count": "total",
            "crowd_correct_mean": "crowd_accuracy",
        }
    )

    logger.info(f"Results:\n\n {results_df}")
    summary_df.to_csv(results_path / f"summary_{time_string}.csv", index=False)

if __name__ == "__main__":
    print(os.getcwd())
    results_path = Path("results")

    csv_file_path = results_path / f"results_240418124558.csv" 

    write_results(csv_file_path)