import os
from inference_L31_context import model_fn, predict_fn, get_probabilities_withcontext
import torch
import pandas as pd

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

suffix_context = '{context}\n"A": {comment}\nDoes response message from "A" seem "{word}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'


output = predict_fn(data = {'input' : [suffix_context.format(word='racist', context='how are you', 
                                                             comment='I hate black people'),
                                      suffix_context.format(word='racist', context='how are you',
                                                            comment = 'we are all the same')],
                                 'use_cache' : True, 
                                 'history' : True, 
                                 'get_history' : False}, 
                            model_and_tokenizer = model_and_tokenizer)




df = pd.read_csv("../adjectives_sortd_en.csv")
special_tokens = df['adjective'].tolist() 

import traceback, pickle


for file in ['../hs_cs/train.csv', '../hs_cs/test.csv']:

    # z = pickle.load(open(f'{file}.pickle', 'rb'))
    distributions = []
    # distributions = z['values']

    df = pd.read_csv(file, sep=',')
    df = df.dropna(subset=['text'])

    df['text+context'] = df['text'] + ' ' + df['context']
    df['slen'] = df['text+context'].map(lambda x: len(x))
    df = df.sort_values(by=['slen'], ascending=True)

    ids = df['id'].tolist()
    mensaje = df['text'].tolist()
    contexto = df['context'].tolist()

    print(f"{bcolors.OKGREEN}Running inference for {len(mensaje)} instances in {file}!{bcolors.ENDC}")
        

    for j, (msg, cont) in tqdm(enumerate(zip(mensaje, contexto)), total = len(mensaje)):
        
        # if ids[j] in z['id']:
        #     continue

        while BATCH_SIZE:
            try:
                probabilities = get_probabilities_withcontext(model_and_tokenizer = model_and_tokenizer,
                                                            special_tokens = special_tokens,
                                                            prompt_format = suffix_context, 
                                                            comment=msg,
                                                            context=cont,
                                        batch_size=BATCH_SIZE)
                distributions += [probabilities]
                break
            except:
                print(traceback.format_exc())
                BATCH_SIZE -= 1
                print(f"{bcolors.FAIL}Batch size {BATCH_SIZE} failed!{bcolors.ENDC}")
                
        if j % 100 == 0 and j:
            with open(f'{file}.pickle', 'wb') as handle:
                pickle.dump({'id':ids[:len(distributions)], 
                            'values': distributions}, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f'{file}.pickle', 'wb') as handle:
        pickle.dump({'id':ids[:len(distributions)], 
                    'values': distributions}, handle, protocol=pickle.HIGHEST_PROTOCOL)

