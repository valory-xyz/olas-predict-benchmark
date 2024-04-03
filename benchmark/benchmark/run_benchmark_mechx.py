#!/usr/bin/env python

import csv
from dotenv import load_dotenv
import json
import os
import pandas as pd
from pathlib import Path
import pickle
from mech_client.interact import interact, ConfirmationType
import time
from tqdm import tqdm
from utils import get_logger, TokenCounterCallback


load_dotenv()
logger = get_logger(__name__)


def prepare_questions(kwargs):
    test_questions = json.load(open("./data/autocast/autocast_questions_filtered.json"))
    with open("./data/autocast/autocast_questions_filtered.pkl", 'rb') as f:
        url_to_content = pickle.load(f)
    num_questions = kwargs.pop("num_questions", len(test_questions))

    questions = []
    for q in test_questions:
        if q["qtype"] == "t/f" and q["answer"] is not None:
            questions.append(q)
        if len(questions) >= num_questions:
            break
    
    return questions, url_to_content

def parse_response(response, test_q):
        request_id = response['requestId']
        result = json.loads(response['result'])
        test_q["p_yes"] = float(result["p_yes"])
        test_q["p_no"] = float(result["p_no"]) 
        cost_dict = response['cost_dict']
        if cost_dict is not None:
            test_q["input_tokens"] = cost_dict["input_tokens"]
            test_q["output_tokens"] = cost_dict["output_tokens"]
            test_q["total_tokens"] = cost_dict["total_tokens"]
            test_q["input_cost"] = cost_dict["input_cost"]
            test_q["output_cost"] = cost_dict["output_cost"]
            test_q["total_cost"] = cost_dict["total_cost"]
        test_q["prompt_response"] = response['prompt'].replace(os.linesep, "")
        metadata = response['metadata']
        if float(result["p_yes"]) == float(result["p_no"]):
            test_q["prediction"] = None
        else:
            test_q["prediction"] = (
                "yes" if test_q["p_yes"] > test_q["p_no"] else "no"
            )
        test_q["Correct"] = test_q["prediction"] == test_q["answer"]
        return test_q

def write_results(csv_file_path):

    results_path = Path(csv_file_path.parent)
    time_string = csv_file_path.stem.split('_', 1)[-1]

    results_df = pd.read_csv(csv_file_path)
    num_errors = results_df['error'].count()
    logger.info(f"Num errors: {str(num_errors)}")
    results_df = results_df.dropna(subset=['prediction'])
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
    questions, url_to_content = prepare_questions(kwargs)
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
            "prompt_response",
            "error",
            "crowd_prediction",
            "crowd_correct"
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
                    "answer": test_question["answer"],
                    "crowd_prediction": test_question['crowd'][-1]['forecast'],
                    "tool": t,
                    "counter_callback": TokenCounterCallback(),
                    "prompt_response": None
                }

                if kwargs["provide_source_links"]:
                    test_q['source_links'] = test_question["source_links"]
                    test_q['source_links'] = {source_link: url_to_content[source_link] for source_link in test_q['source_links']}

                crowd_forecast = test_question['crowd'][-1]['forecast']
                test_q["crowd_prediction"] = "yes" if crowd_forecast > 0.5 else "no" if crowd_forecast < 0.5 else None
                test_q["crowd_correct"] = test_q['crowd_prediction'] == test_q["answer"]


                CURRENT_RETRIES = 0

                while True:
                    try:
                        response = interact(
                            prompt=test_q['prompt'],
                            agent_id=6,
                            tool=t,
                            confirmation_type=ConfirmationType.ON_CHAIN,
                            extra_attributes={'model': 'gpt-4-0125-preview', "source_links":test_q["source_links"]}
                            # private_key_path="ethereum_private_key.txt"
                            )
                        test_q = parse_response(response, test_q)
                        break

                    except Exception as e:
                        logger.error(f"Error running benchmark for tool {t}: {e}")
                        test_q["error"] = e
                        break

                if kwargs["provide_source_links"]:
                    del test_q["source_links"]
                del test_q["counter_callback"]
                del test_q["prompt_response"]

                writer.writerow(test_q)

    write_results(csv_file_path)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total Time: {total_time} seconds")


if __name__ == "__main__":
    kwargs = {}
    # kwargs["num_questions"] = 1
    kwargs["tools"] = [
        # "prediction-online",
        # "prediction-offline",
        # "prediction-online-summarized-info",
        "prediction-offline-sme",
        # "prediction-online-sme",
        # "claude-prediction-offline",
        # "claude-prediction-online",
        # 'prediction-request-rag',
        # "prediction-with-research-conservative",
        # "prediction-with-research-bold",
    ]
    kwargs["provide_source_links"] = True

    run_benchmark(kwargs)
