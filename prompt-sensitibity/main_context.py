import os
from inference_L31_no_context import model_fn, predict_fn, get_probabilities_nocontext, get_history_fn
import torch
import pandas as pd
import traceback, pickle

from tqdm import tqdm

import os, sys, dotenv
sys.path.append('..')
dotenv.load_dotenv()

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


persona = ["You are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.",
"You are an expert in social psychology.",
"",
"You are a Linguist.",
# "You are a computer scientist.",
"You are a content moderator.",
"You are a psychologist.",
"You are a social media expert.",
"You are a political scientist.",
"You are a sociologist.",]
# # "You are a bot."]



repository = "meta-llama/Llama-3.1-8B-Instruct"#
model_id=repository.split("/")[-1]

os.system(f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} /workspace/{model_id}')

BATCH_SIZE = 244
model_and_tokenizer = model_fn(f"/workspace/{model_id}")
prefix = 'Consider what a person "A" states when racting to a "CONTEXT":\n\n"CONTEXT": ' 


history = get_history_fn(model_and_tokenizer = model_and_tokenizer,
                         prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{persona[0]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prefix}""",
                         history_size = BATCH_SIZE)


suffix_context = '{context}\n"A": {comment}\nDoes response message from "A" seem "{placeholder}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

output = predict_fn(data = {'input' : [suffix_nocontext.format(placeholder='racist',
                                                             sentence='Ich hasse schwarze Menschen'),
                                      suffix_nocontext.format(placeholder='racist',
                                                            sentence = 'Wir sind alle gleich')],
                                 'use_cache' : True, 
                                 'history' : True, 
                                 'get_history' : False}, 
                            model_and_tokenizer = model_and_tokenizer, 
                            history = history)

print(output)

df = pd.read_csv("../adjectives_sortd_en.csv")
special_tokens = df['adjective'].tolist()

for file in ['train.csv', 'test.csv']:

    values = {}
    for k in range(len(persona)):

        df = pd.read_csv(file, sep=',')
        ids = df['id'].tolist()
        sentences = df['text'].tolist()

        print(f"{bcolors.OKGREEN}Running inference for {len(sentences)} instances in persona {k}!{bcolors.ENDC}")

        history = get_history_fn(model_and_tokenizer = model_and_tokenizer,
                                prefix = persona[k],
                                history_size = BATCH_SIZE)
        
        distributions = []
        for j, sente in tqdm(enumerate(sentences), total = len(sentences)):
            
            while BATCH_SIZE:
                try:
                    
                    probabilities = get_probabilities_nocontext(model_and_tokenizer = model_and_tokenizer,
                                                                special_tokens = special_tokens,
                                                                prompt_format = suffix_nocontext, 
                                                                sentence = sente,
                                                                batch_size=BATCH_SIZE,
                                                                history = history)
                    distributions += [probabilities]
                    break
                except:
                    print(traceback.format_exc())
                    BATCH_SIZE -= 1
                    print(f"{bcolors.FAIL}Batch size {BATCH_SIZE} failed!{bcolors.ENDC}")
                    
        values[k] = {'id':ids[:len(distributions)], 
                        'text':sentences[:len(distributions)],
                        'values': distributions}
        
        with open(f'diferent_persona_vectors{file}.pickle', 'wb') as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)

