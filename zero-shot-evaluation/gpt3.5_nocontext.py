#%%
import pandas as pd
import openai, json
import pandas as pd
from tqdm import tqdm
import traceback, pickle

import os, sys, dotenv
dotenv.load_dotenv()

client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))


def get_response( message , context = None): 
  
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      # logprobs = True,
      messages=[
        {"role": "system", "content": "You are an expert in social psychology."},
        {"role": "user", "content": f'This mensaje is "counterspeech" or "non-counterspeech":\n{message}. Only output "counterspeech" or "non-counterspeech".'},
      ],
       max_tokens = 10,

    )
    # print(response.choices[0].message.content)
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
    
def majority_vote(message, votes_num = 1):
    o = []
    votes = []
    for i in range(votes_num):
        
        z = get_response(message), 1
        if z[0] != "unknown":
            votes += [z[0]]
        o += [z[1]]
        
    if len(votes) == 0:
        votes += ["unknown"]
    return votes[0], np.mean(o)



#%%

FILE_PATH = "../tsnh/TSNH_uniform.csv"

df = pd.read_csv(FILE_PATH, sep=',')
ids = df['id'].tolist()

messages = df['text'].tolist()


runs = []
for i in range(4):

    print(f"{bcolors.OKGREEN}Running inference for {len(messages)} instances!{bcolors.ENDC}")

    itera = tqdm(enumerate(messages), total = len(messages))
    runs += [{'id':[], 'values': []}]
    for j, mess in itera:
        
        probabilities, mean_intents = majority_vote(message = mess)
        runs[-1]['values'] += [probabilities]
        runs[-1]['id'] += [ids[j]]

        itera.set_postfix(mean_intents=mean_intents)
        
    with open(f'{FILE_PATH}.gpt35-turbo.pickle', 'wb') as handle:
        pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)