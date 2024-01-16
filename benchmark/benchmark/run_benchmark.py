#!/usr/bin/env python

from dotenv import load_dotenv
import json
import os
import pandas as pd
from prediction_request import tool
from tqdm import tqdm

load_dotenv()

def run_benchmark(kwargs):
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""

    test_questions = json.load(open('./data/autocast/autocast_questions.json'))

    results_df = pd.DataFrame()

    for i in tqdm(range(10)):
        test_question = test_questions[i]

        qtype = test_question['qtype']
        if qtype != "t/f": # only use true/false question types
            print("Not a true/false question. Skipping...")
            continue

        test_q = {
            'prompt': test_question['question'], 
            'source_links': test_question['source_links'], 
            'answer': test_question['answer']
        }

        response = tool.run(**{**test_q, **kwargs})

        result = json.loads(response[0])
        test_q['p_yes'] = float(result['p_yes'])
        test_q['p_no'] = float(result['p_no'])
        test_q['prompt_response'] = response[1].replace(os.linesep, "")

        if float(result['p_yes']) == float(result['p_no']):
            test_q['prediction'] = "undecided"
        else:
            test_q['prediction'] = 'yes' if test_q['p_yes'] > test_q['p_no'] else 'no'

        test_q['Correct'] = (test_q['prediction'] == test_q['answer'])

        results_df = pd.concat([results_df, pd.DataFrame([test_q]).drop(['source_links', 'prompt_response'], axis=1)])

    results_df.to_csv('results.csv', index=False)
    accuracy = (results_df['Correct'].sum() / len(results_df)) 

    print(results_df)
    print('Accuracy: ', accuracy)

if __name__ == "__main__":
    kwargs = {}
    kwargs["tool"] = "prediction-online"
    kwargs["api_keys"] = {}
    kwargs["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
    kwargs["num_urls"] = 2

    run_benchmark(kwargs)