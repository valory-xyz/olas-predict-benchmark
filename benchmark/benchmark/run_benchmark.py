#!/usr/bin/env python

from dotenv import load_dotenv
import json
import os
import pandas as pd
from prediction_request import tool as prediction_request_tool
from prediction_request_sme import tool as prediction_request_sme_tool
from tqdm import tqdm
from benchmark.utils import get_logger

load_dotenv()
logger = get_logger(__name__)


def tool_map(tool):
    """Map the tool name to the tool class."""

    if tool in [
        "prediction-online",
        "prediction-offline",
        "prediction-online-summarized-info",
    ]:
        return prediction_request_tool
    elif tool in ["prediction-offline-sme", "prediction-online-sme"]:
        return prediction_request_sme_tool
    else:
        raise Exception(f"Tool {tool} not found.")


def run_benchmark(kwargs):
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""

    logger.info("Running benchmark tests...")

    test_questions = json.load(open("./data/autocast/autocast_questions.json"))

    results_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    tools = kwargs.pop("tools")
    num_questions = kwargs.pop("num_questions", 10)

    # Prepare the questions
    questions = []

    for q in test_questions:
        if q["qtype"] == "t/f":
            questions.append(q)

        if len(questions) >= num_questions:
            break

    logger.info(f"Running {num_questions} questions for each tool: {tools}")

    for t in tools:
        for test_question in tqdm(
            questions, desc=f"Running tool {t}", total=len(questions)
        ):
            test_q = {
                "prompt": test_question["question"],
                "source_links": test_question["source_links"],
                "answer": test_question["answer"],
                "tool": t,
            }

            try:
                tool = tool_map(t)
                response = tool.run(**{**test_q, **kwargs})
                result = json.loads(response[0])
                test_q["p_yes"] = float(result["p_yes"])
                test_q["p_no"] = float(result["p_no"])
                test_q["prompt_response"] = response[1].replace(os.linesep, "")

                if float(result["p_yes"]) == float(result["p_no"]):
                    test_q["prediction"] = "undecided"
                else:
                    test_q["prediction"] = (
                        "yes" if test_q["p_yes"] > test_q["p_no"] else "no"
                    )

                test_q["Correct"] = test_q["prediction"] == test_q["answer"]

                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame([test_q]).drop(
                            ["source_links", "prompt_response"], axis=1
                        ),
                    ]
                )

            except Exception as e:
                logger.error(f"Error running benchmark for tool {t}: {e}")

        tool_df = results_df[results_df["tool"] == t]
        accuracy = tool_df["Correct"].mean()
        num_correct = tool_df["Correct"].sum()
        num_total = tool_df["Correct"].count()

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "tool": t,
                            "accuracy": accuracy,
                            "num_correct": num_correct,
                            "num_total": num_total,
                        }
                    ]
                ),
            ]
        )
        logger.info(f"Tool {t} accuracy: {accuracy}")

    logger.info("Benchmark tests complete.")
    logger.info(f"Results:\n\n {results_df}")
    results_df.to_csv("results.csv", index=False)
    summary_df.to_csv("summary.csv", index=False)


if __name__ == "__main__":
    kwargs = {}
    kwargs["num_questions"] = 4
    kwargs["tools"] = [
        "prediction-online",
        "prediction-offline",
        "prediction-online-summarized-info",
        "prediction-offline-sme",
        "prediction-online-sme",
    ]
    kwargs["api_keys"] = {}
    kwargs["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
    kwargs["num_urls"] = 2

    run_benchmark(kwargs)
