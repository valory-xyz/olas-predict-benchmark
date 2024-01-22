#!/usr/bin/env python

import csv
from dotenv import load_dotenv
import json
import os
import openai
import pandas as pd
from pathlib import Path
from prediction_request import prediction_request
from prediction_request_sme import prediction_request_sme
from prediction_request_claude import prediction_request_claude
import time
from tqdm import tqdm
from utils import get_logger, TokenCounterCallback

load_dotenv()
logger = get_logger(__name__)


def tool_map(tool):
    """Map the tool name to the tool class."""

    if tool in [
        "prediction-online",
        "prediction-offline",
        "prediction-online-summarized-info",
    ]:
        return prediction_request
    elif tool in ["prediction-offline-sme", "prediction-online-sme"]:
        return prediction_request_sme
    elif tool in ["claude-prediction-offline", "claude-prediction-online"]:
        return prediction_request_claude
    else:
        raise Exception(f"Tool {tool} not found.")


def run_benchmark(kwargs):
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""

    logger.info("Running benchmark tests...")
    start_time = time.time()

    test_questions = json.load(open("./data/autocast/autocast_questions.json"))

    tools = kwargs.pop("tools")
    num_questions = kwargs.pop("num_questions", len(test_questions))
    MAX_RETRIES = kwargs.pop("max_retries", 3)

    # Prepare the questions
    questions = []

    for q in test_questions:
        if q["qtype"] == "t/f":
            questions.append(q)

        if len(questions) >= num_questions:
            break

    logger.info(f"Running {num_questions} questions for each tool: {tools}")

    results_path = Path("results")
    if not results_path.exists():
        results_path.mkdir(exist_ok=True)

    time_string = time.strftime("%y%m%d%H%M%S", time.localtime(start_time))
    csv_file_path = results_path / f"results_{time_string}.csv"

    with open(csv_file_path, mode="a", newline="") as file:
        fieldnames = [
            "prompt",
            "answer",
            "tool",
            "p_yes",
            "p_no",
            "prediction",
            "Correct",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        for t in tools:
            for test_question in tqdm(
                questions, desc=f"Running tool {t}", total=len(questions)
            ):
                test_q = {
                    "prompt": test_question["question"],
                    "source_links": test_question["source_links"],
                    "answer": test_question["answer"],
                    "tool": t,
                    "counter_callback": TokenCounterCallback(),
                }

                CURRENT_RETRIES = 0

                while True:
                    try:
                        tool = tool_map(t)
                        response = tool.run(**{**test_q, **kwargs})
                        result = json.loads(response[0])
                        test_q["p_yes"] = float(result["p_yes"])
                        test_q["p_no"] = float(result["p_no"])
                        if response[2] is not None:
                            test_q["input_tokens"] = response[2].input_tokens
                            test_q["output_tokens"] = response[2].output_tokens
                            test_q["total_tokens"] = response[2].total_tokens
                            test_q["input_cost"] = response[2].input_cost
                            test_q["output_cost"] = response[2].output_cost
                            test_q["total_cost"] = response[2].total_cost
                        # test_q["prompt_response"] = response[1].replace(os.linesep, "")

                        if float(result["p_yes"]) == float(result["p_no"]):
                            test_q["prediction"] = "undecided"
                        else:
                            test_q["prediction"] = (
                                "yes" if test_q["p_yes"] > test_q["p_no"] else "no"
                            )

                        test_q["Correct"] = test_q["prediction"] == test_q["answer"]

                        del test_q["source_links"]
                        del test_q["counter_callback"]

                        writer.writerow(test_q)

                        break

                    except openai.error.OpenAIError as e:
                        logger.error(f"Error running benchmark for tool {t}: {e}")
                        CURRENT_RETRIES += 1
                        if CURRENT_RETRIES > MAX_RETRIES:
                            logger.error(
                                f"Max retries reached for tool {t}. Skipping question."
                            )
                            break
                        else:
                            logger.info(
                                f"Retrying tool {t} for question {test_q['prompt']}"
                            )

                    except Exception as e:
                        logger.error(f"Error running benchmark for tool {t}: {e}")
                        break

    results_df = pd.read_csv(csv_file_path)
    grouped_df = results_df.groupby("tool").agg(
        {
            "Correct": ["mean", "sum", "count"],
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
        }
    )

    logger.info("Benchmark tests complete.")
    logger.info(f"Results:\n\n {results_df}")
    summary_df.to_csv(results_path / f"summary_{time_string}.csv", index=False)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Time: {total_time} seconds")


if __name__ == "__main__":
    kwargs = {}
    # kwargs["num_questions"] = 4
    kwargs["tools"] = [
        "prediction-online",
        "prediction-offline",
        "prediction-online-summarized-info",
        "prediction-offline-sme",
        "prediction-online-sme",
        "claude-prediction-offline",
        "claude-prediction-online",
    ]
    kwargs["api_keys"] = {}
    kwargs["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
    kwargs["api_keys"]["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
    kwargs["num_urls"] = 3
    kwargs["num_words"] = 300

    run_benchmark(kwargs)
