import pandas as pd
import openai, pickle, os, dotenv
from tqdm import tqdm
import traceback
from fire import Fire

from typing import Literal

# Load environment variables
dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def get_response(comment, classes, model, context=None):
    if context is not None and context != '':
        prompt = (
            'Consider what a person "A" states when reacting to a "CONTEXT":\n\n'
            f'"CONTEXT": {context}\n"A": {comment}\n'
            f'Does response message from "A" seem to be one of [{", ".join(classes)}]? '
            'Answer in a single word.'
        )
    else:
        prompt = (
            f'Consider what a person "A" states: "A": {comment}\n'
            f'Does "A" seem to be {", ".join(classes[:-1]) + ", or " + classes[-1]}. '
            'Answer with a single word.'
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content


def main(file_path: str, model: Literal["gpt-3.5-turbo", "chatgpt-4o-latest"]):
    df = pd.read_csv(file_path, sep=',')
    ids = df['id'].tolist()
    comment = df['text'].tolist()
    classes = sorted(list(set(df['Class'].values))) 
    
    context = df['context'].tolist() if 'context' in df.columns else [''] * len(comment)

    runs = []
    for _ in range(4):
        print(f"{bcolors.OKGREEN}Running inference for {len(comment)} instances!{bcolors.ENDC}")
        itera = tqdm(enumerate(zip(context, comment)), total=len(context))
        runs += [{'id': [], 'values': []}]
        for j, (cont, com) in itera:

            try:
                if 'context' in df.columns:
                    prediction = get_response(comment=com, context=cont, classes=classes, model=model)
                else:
                    prediction = get_response(comment=com, classes=classes, model=model)
            except Exception:
                prediction = "error"
                traceback.print_exc()
            runs[-1]['values'] += [prediction]
            runs[-1]['id'] += [ids[j]]
        with open(f'{file_path}.{model}.pickle', 'wb') as handle:
            pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    Fire(main)
