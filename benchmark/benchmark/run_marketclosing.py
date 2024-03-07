#!/usr/bin/env python

import csv
from dotenv import load_dotenv
import json
from mech.packages.valory.customs.resolve_market import resolve_market
from mech.packages.napthaai.customs.resolve_market_reasoning import (
    resolve_market_reasoning,
)
import os
import openai
import pandas as pd
from pathlib import Path
import pickle
import re
import time
from tqdm import tqdm
from utils import get_logger, TokenCounterCallback

load_dotenv()
logger = get_logger(__name__)


def tool_map(tool):
    """Map the tool name to the tool class."""

    if tool in ["resolve-market"]:
        return resolve_market
    elif tool in [
        "resolve-market-reasoning-gpt-3.5-turbo",
        "resolve-market-reasoning-gpt-4",
    ]:
        return resolve_market_reasoning
    else:
        raise Exception(f"Tool {tool} not found.")


def prepare_questions(kwargs):
    test_questions = json.load(open("./data/autocast/autocast_questions_filtered.json"))
    num_questions = kwargs.pop("num_questions", len(test_questions))
    questions = []
    for q in test_questions:
        if q["qtype"] == "t/f" and q["answer"] is not None:
            questions.append(q)
        if len(questions) >= num_questions:
            break
    return questions


def parse_response(response, test_q):
    result = response[0]

    test_q["prediction"] = None
    test_q["reasoning"] = None
    test_q["queries"] = None
    test_q["Correct"] = None

    if not result:
        return test_q

    if result.dict().get("has_occurred") == True:
        test_q["prediction"] = "yes"
    elif result.dict().get("has_occurred") == False:
        test_q["prediction"] = "no"
    elif result.dict().get("is_valid") == False:
        test_q["prediction"] = "invalid"
        return test_q
    elif result.dict().get("is_determinable") == False:
        test_q["prediction"] = "indeterminable"
    else:
        test_q["prediction"] = None

    test_q["reasoning"] = response[1].replace(os.linesep, "")
    test_q["prompt_response"] = response[2].replace(os.linesep, "")
    test_q["queries"] = response[3]

    if test_q["prediction"] in ("yes", "no"):
        test_q["Correct"] = test_q["prediction"] == test_q["answer"]
    else:
        test_q["Correct"] = None

    if response[4] is not None:
        test_q["input_tokens"] = response[4].cost_dict["input_tokens"]
        test_q["output_tokens"] = response[4].cost_dict["output_tokens"]
        test_q["total_tokens"] = response[4].cost_dict["total_tokens"]
        test_q["input_cost"] = response[4].cost_dict["input_cost"]
        test_q["output_cost"] = response[4].cost_dict["output_cost"]
        test_q["total_cost"] = response[4].cost_dict["total_cost"]

    print("Reasoning: ", test_q["reasoning"])
    print("Prediction/Correct:", test_q["prediction"], ", ", test_q["Correct"])

    return test_q


def write_results(csv_file_path):
    results_path = Path(csv_file_path.parent)
    time_string = csv_file_path.stem.split("_", 1)[-1]

    results_df = pd.read_csv(csv_file_path)
    results_df = results_df.dropna(subset=["prediction"])
    grouped_df = results_df.groupby("tool").agg(
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


def run_benchmark(kwargs):
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""

    logger.info("Running benchmark tests...")

    tools = kwargs.pop("tools")
    MAX_RETRIES = kwargs.pop("max_retries", 3)
    questions = prepare_questions(kwargs)
    logger.info(f"Running {len(questions)} questions for each tool: {tools}")

    results_path = Path("results")
    if not results_path.exists():
        results_path.mkdir(exist_ok=True)

    start_time = time.time()
    time_string = time.strftime("%y%m%d%H%M%S", time.localtime(start_time))
    csv_file_path = results_path / f"results_{time_string}.csv"

    with open(csv_file_path, mode="a", newline="") as file:
        fieldnames = [
            "prompt",
            "answer",
            "tool",
            "prediction",
            "Correct",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
            "prompt_response",
            "crowd_prediction",
            "crowd_correct",
            "queries",
            "dates",
            "reasoning",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        for t in tools:
            correct_answers = 0
            total_answers = 0
            for test_question in tqdm(
                questions, desc=f"Running tool {t}", total=len(questions)
            ):
                print("Question: ", test_question["question"])
                test_q = {
                    "prompt": test_question["question"],
                    "answer": test_question["answer"],
                    "crowd_prediction": test_question["crowd"][-1]["forecast"],
                    "tool": t,
                    "counter_callback": TokenCounterCallback(),
                    "prompt_response": None,
                }

                crowd_forecast = test_question["crowd"][-1]["forecast"]
                test_q["crowd_prediction"] = (
                    "yes"
                    if crowd_forecast > 0.5
                    else "no"
                    if crowd_forecast < 0.5
                    else None
                )
                test_q["crowd_correct"] = test_q["crowd_prediction"] == test_q["answer"]

                CURRENT_RETRIES = 0

                while True:
                    try:
                        tool = tool_map(t)
                        response = tool.run(**{**test_q, **kwargs})
                        test_q = parse_response(response, test_q)
                        if test_q["Correct"] == True:
                            correct_answers += 1
                        if test_q["prediction"] is not None:
                            total_answers += 1

                        print(
                            f"===========ACCURACY============== {correct_answers/total_answers*100}%"
                        )
                        break

                    except openai.APIError as e:
                        logger.error(f"Error running benchmark for tool {t}: {e}")
                        CURRENT_RETRIES += 1
                        if CURRENT_RETRIES > MAX_RETRIES:
                            logger.error(
                                f"Max retries reached for tool {t}. Skipping question."
                            )
                            test_q["error"] = e
                            break
                        else:
                            logger.info(
                                f"Retrying tool {t} for question {test_q['prompt']}"
                            )
                            continue

                    except Exception as e:
                        logger.error(f"Error running benchmark for tool {t}: {e}")
                        test_q["error"] = e
                        break

                del test_q["counter_callback"]

                writer.writerow(test_q)

    write_results(csv_file_path)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Time: {total_time} seconds")


if __name__ == "__main__":
    kwargs = {}
    kwargs["num_questions"] = 5
    kwargs["tools"] = [
        # "resolve-market",
        "resolve-market-reasoning-gpt-3.5-turbo",
        # "resolve-market-reasoning-gpt-4",
    ]
    kwargs["api_keys"] = {}
    kwargs["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
    kwargs["api_keys"]["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
    kwargs["api_keys"]["google_api_key"] = os.getenv("google_api_key")
    kwargs["api_keys"]["google_engine_id"] = os.getenv("google_engine_id")
    kwargs["api_keys"]["newsapi"] = os.getenv("NEWS_API_KEY")
    kwargs["num_urls"] = 3
    kwargs["num_words"] = 300
    run_benchmark(kwargs)
