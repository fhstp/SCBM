import os
from inference_L31 import model_fn, predict_fn, get_probabilities_nocontext, get_history_fn
import torch
import pandas as pd
import traceback, pickle

from tqdm import tqdm
from fire import Fire

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
"You are a content moderator.",
"You are a psychologist.",
"You are a social media expert.",
"You are a political scientist.",
"You are a sociologist."]



repository = "meta-llama/Llama-3.1-8B-Instruct"#
model_id=repository.split("/")[-1]

os.system(f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} /workspace/{model_id}')

BATCH_SIZE = 244
model_and_tokenizer = model_fn(f"/workspace/{model_id}")
history = get_history_fn(model_and_tokenizer = model_and_tokenizer,
                         prefix = persona[0],
                         history_size = BATCH_SIZE)

# Default (no-context) suffix used for quick smoke test below and for datasets without context
suffix_nocontext = '"A": {sentence}\nDoes "A" seem "{placeholder}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

# Quick smoke test to validate model wiring
_test_output = predict_fn(
    data={
        'input': [
            suffix_nocontext.format(placeholder='racist', sentence='Ich hasse schwarze Menschen'),
            suffix_nocontext.format(placeholder='racist', sentence='Wir sind alle gleich')
        ],
        'use_cache': True,
        'history': True,
        'get_history': False
    },
    model_and_tokenizer=model_and_tokenizer,
    history=history
)
print(_test_output)


def get_probabilities_with_context(model_and_tokenizer,
                                   special_tokens,
                                   prompt_format,
                                   comment,
                                   context,
                                   batch_size,
                                   history=None):
    """Compute probabilities for context-aware prompts, mirroring no-context batching logic."""

    sorted_instances = [
        (prompt_format.format(placeholder=adj, comment=comment, context=context), idx)
        for idx, adj in enumerate(special_tokens)
    ]
    sorted_instances = sorted(sorted_instances, key=lambda x: -len(x[0]))

    probabilities = None
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader([i[0] for i in sorted_instances], batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            output = predict_fn(
                {'input': batch, 'use_cache': True},
                model_and_tokenizer=model_and_tokenizer,
                history=history
            )
            probs = output['probs'].clone().detach()
            probabilities = torch.cat([probabilities, probs]) if probabilities is not None else probs
        del dataloader

    values = sorted([(x.item(), y[-1]) for x, y in zip(probabilities, sorted_instances)], key=lambda x: x[1])
    return [x[0] for x in values]

def main(adjectives_path: str = "../adjectives_sortd_en.csv",
         data_path: str = "../Tasks/germeval/train.csv",
         output_path: str = "../Tasks/germeval/"):
    
    """
    Main function to compute the representations for all the adjectives in the list
    using different personas to evaluate the impact of the persona on the representations and
    the prompt sensitivity.

    :param adjectives_path: Path to the csv file containing the list of adjectives
    :param data_path: Path to the csv file containing the data to be processed
    :param output_path: Path to the output folder where the representations will be saved
    """

    df_adj = pd.read_csv(adjectives_path)
    special_tokens = df_adj['adjective'].tolist()

    values = {}
    for k in range(len(persona)):
        df_data = pd.read_csv(data_path, sep=',')
        ids = df_data['id'].tolist()
        sentences = df_data['text'].tolist()
        use_context = 'context' in df_data.columns

        if use_context:
            contexts = df_data['context'].fillna('').tolist()
            # Build history with system + user prefix that introduces the CONTEXT header
            prefix_user = 'Consider what a person "A" states when racting to a "CONTEXT":\n\n"CONTEXT": '
            history = get_history_fn(
                model_and_tokenizer=model_and_tokenizer,
                prefix=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{persona[k]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prefix_user}""",
                history_size=BATCH_SIZE
            )
            prompt_format = '{context}\n"A": {comment}\nDoes response message from "A" seem "{placeholder}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        else:
            history = get_history_fn(
                model_and_tokenizer=model_and_tokenizer,
                prefix=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{persona[k]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n""",
                history_size=BATCH_SIZE
            )
            prompt_format = suffix_nocontext

        print(f"{bcolors.OKGREEN}Running inference for {len(sentences)} instances in persona {k}!{bcolors.ENDC}")

        distributions = []
        for j in tqdm(range(len(sentences)), total=len(sentences)):
            current_batch_size = BATCH_SIZE
            while current_batch_size:
                try:
                    if use_context:
                        probabilities = get_probabilities_with_context(
                            model_and_tokenizer=model_and_tokenizer,
                            special_tokens=special_tokens,
                            prompt_format=prompt_format,
                            comment=sentences[j],
                            context=contexts[j],
                            batch_size=current_batch_size,
                            history=history
                        )
                    else:
                        probabilities = get_probabilities_nocontext(
                            model_and_tokenizer=model_and_tokenizer,
                            special_tokens=special_tokens,
                            prompt_format=prompt_format,
                            sentence=sentences[j],
                            batch_size=current_batch_size,
                            history=history
                        )
                    distributions.append(probabilities)
                    break
                except Exception:
                    print(traceback.format_exc())
                    current_batch_size -= 1
                    print(f"{bcolors.FAIL}Batch size {current_batch_size} failed!{bcolors.ENDC}")

        values[k] = {
            'id': ids[:len(distributions)],
            'text': sentences[:len(distributions)],
            'values': distributions
        }

    with open(os.path.join(output_path, 'diferent_persona_vectors.pickle'), 'wb') as handle:
        pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    Fire(main)

