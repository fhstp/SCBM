import os
from inference_L31_no_context import model_fn, predict_fn, get_probabilities_nocontext
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


repository = "meta-llama/Llama-3.1-8B-Instruct"#
model_id=repository.split("/")[-1]

if not os.path.exists(f'/workspace/{model_id}'):
    os.system(f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} /workspace/{model_id}')

BATCH_SIZE = 244
model_and_tokenizer = model_fn(f"/workspace/{model_id}", history_size = BATCH_SIZE)

suffix_nocontext = '"A": {sentence}\n\n{definition}. Does "A" seem "{placeholder}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
# suffix_germeval = '"{placeholder}" den folgenden Satz in irgendeiner Weise beschreibt: "{sentence}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

output = predict_fn(data = {'input' : [suffix_nocontext.format(placeholder='racist',
                                                             sentence='Ich hasse schwarze Menschen',
                                                             definition='Rassismus ist eine Ideologie, die Menschen aufgrund ihrer ethnischen Zugehörigkeit oder Hautfarbe diskriminiert.'),
                                      suffix_nocontext.format(placeholder='racist',
                                                              definition='Rassismus ist eine Ideologie, die Menschen aufgrund ihrer ethnischen Zugehörigkeit oder Hautfarbe diskriminiert.',
                                                            sentence = 'Wir sind alle gleich')],
                                 'use_cache' : True, 
                                 'history' : True, 
                                 'get_history' : False}, 
                            model_and_tokenizer = model_and_tokenizer)



df = pd.read_csv("../Bootleneck-ICL/adjectives_with_definitions.csv")
# df = pd.read_csv("adjectives_sortd_en.csv")

special_tokens = df['adjective'].tolist()
definitions = df['definitions'].tolist()

for file in ['../tsnh/TSNH_uniform.csv']:
    distributions = []

    df = pd.read_csv(file, sep=',')
    # drop rows with empty text
    df = df.dropna(subset=['text'])
    # get the largest element in text

    df['slen'] = df['text'].map(lambda x: len(x))
    df = df.sort_values(by=['slen'], ascending=True)[:len(df) // 3]

    ids = df['id'].tolist()
    sentences = df['text'].tolist()

    print(f"{bcolors.OKGREEN}Running inference for {len(sentences)} instances in {file}!{bcolors.ENDC}")
        
    sentences = [s for s in sentences if len(s) > 0]

    for j, sente in tqdm(enumerate(sentences), total = len(sentences)):
        
        while BATCH_SIZE:
            try:
                
                probabilities = get_probabilities_nocontext(model_and_tokenizer = model_and_tokenizer,
                                                            special_tokens = special_tokens,
                                                            tokens_definitions = definitions,
                                                            prompt_format = suffix_nocontext, 
                                                            sentence = sente,
                                        batch_size=BATCH_SIZE)
                distributions += [probabilities]
                break
            except:
                print(traceback.format_exc())
                BATCH_SIZE -= 1
                print(f"{bcolors.FAIL}Batch size {BATCH_SIZE} failed!{bcolors.ENDC}")
                
        if j % 100 == 0 and j:
            with open(f'{file}-icl-adj.pickle', 'wb') as handle:
                pickle.dump({'id':ids[:len(distributions)], 
                            'text':sentences[:len(distributions)],
                            'values': distributions}, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f'{file}-icl-adj.pickle', 'wb') as handle:
        pickle.dump({'id':ids[:len(distributions)], 
                    'text':sentences[:len(distributions)],
                    'values': distributions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

