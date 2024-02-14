#!/usr/bin/env python

import csv
from dotenv import load_dotenv
import json
from mech.tools.resolve_market import resolve_market
from mech.tools.resolve_market_reasoning import resolve_market_reasoning
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
    elif tool in ["resolve-market-reasoning"]:
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
        result = json.loads(response[0])

        if result["has_occurred"] == True:
            test_q["prediction"] = "yes"
        elif result["has_occurred"] == False:
            test_q["prediction"] = "no"
        elif result["has_occurred"] == None:
            test_q["prediction"] = None
        test_q['reasoning'] = result['reasoning']

        test_q["prompt_response"] = response[1].replace(os.linesep, "")
        test_q["queries"] = response[2]
        test_q["dates"] = response[3]

        if test_q["prediction"] is not None:
            test_q["Correct"] = test_q["prediction"] == test_q["answer"]
        else:
            test_q["Correct"] = None
        return test_q

def write_results(csv_file_path):

    results_path = Path(csv_file_path.parent)
    time_string = csv_file_path.stem.split('_', 1)[-1]

    results_df = pd.read_csv(csv_file_path)
    results_df = results_df.dropna(subset=['prediction'])
    grouped_df = results_df.groupby("tool").agg(
        {
            "Correct": ["mean", "sum", "count"],
            "crowd_correct": ["mean"],
        }
    )

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]
    summary_df = grouped_df.reset_index().rename(
        columns={
            "Correct_mean": "accuracy",
            "Correct_sum": "correct",
            "Correct_count": "total",
            "crowd_correct_mean": "crowd_accuracy"
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
            "question",
            "answer",
            "tool",
            "prediction",
            "Correct",
            "prompt_response",
            "crowd_prediction",
            "crowd_correct",
            "queries",
            "dates",
            "reasoning"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        for t in tools:
            for test_question in tqdm(
                questions, desc=f"Running tool {t}", total=len(questions)
            ):
                test_q = {
                    "question": test_question["question"],
                    "answer": test_question["answer"],
                    "crowd_prediction": test_question['crowd'][-1]['forecast'],
                    "tool": t,
                    "counter_callback": TokenCounterCallback(),
                    "prompt_response": None
                }

                crowd_forecast = test_question['crowd'][-1]['forecast']
                test_q["crowd_prediction"] = "yes" if crowd_forecast > 0.5 else "no" if crowd_forecast < 0.5 else None
                test_q["crowd_correct"] = test_q['crowd_prediction'] == test_q["answer"]

                CURRENT_RETRIES = 0

                while True:
                    try:
                        tool = tool_map(t)
                        response = tool.run(**{**test_q, **kwargs})
                        test_q = parse_response(response, test_q)
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
                                f"Retrying tool {t} for question {test_q['question']}"
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
        "resolve-market-reasoning",
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
