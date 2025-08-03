#%%
import pandas as pd
import openai, json
import pandas as pd
from tqdm import tqdm
import traceback, pickle

import os, sys, dotenv
dotenv.load_dotenv()
client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))


def get_response( comment , context ): 
    

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      # logprobs = True,
      messages=[
        {"role": "system", "content": "You are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers."},
        {"role": "user", "content": f'Consider what a person "A" states when reacting to a "CONTEXT":\n\n"CONTEXT": {context}\n"A": {comment}\nDoes response message from "A" seem to be one of [denouncing, hypocrisy, question, unrelated, humor, facts, support]?. Answer in a single word with: "denouncing", "hypocrisy", "question", "unrelated", "humor", "facts", "support".'},
      ]
    )
    
    return response.choices[0].message.content


    
import json
import numpy as np
import traceback

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

#%%

FILE_PATH = "../conan/test.csv"


df = pd.read_csv(FILE_PATH, sep=',')
ids = df['id'].tolist()

comment = df['text'].tolist()
context = df['context'].tolist()


runs = []
for i in range(4):

    print(f"{bcolors.OKGREEN}Running inference for {len(comment)} instances!{bcolors.ENDC}")

    itera = tqdm(enumerate(zip(context, comment)), total = len(context))
    runs += [{'id':[], 'values': []}]
    for j, (cont, com) in itera:
        
        probabilities = get_response(comment = cont, context = com)
        # print(probabilities)
        runs[-1]['values'] += [probabilities]
        runs[-1]['id'] += [ids[j]]
        
    with open(f'{FILE_PATH}.gpt35.pickle', 'wb') as handle:
        pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

