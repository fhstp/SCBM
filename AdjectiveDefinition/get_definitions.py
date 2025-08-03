from pathlib import Path
import pandas as pd
from tqdm import tqdm

import fire

import os, dotenv, sys
import openai

sys.path.append('..')
dotenv.load_dotenv()

client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def get_response( message ): 
  
    response = client.chat.completions.create(
      model="chatgpt-4o-latest",
      # logprobs = True,
      messages=[
        {"role": "user", "content": message},
      ],
    )

    return response.choices[0].message.content

def main(adjectives_path: Path = 'adjectives_sortd_en.csv'):

    df = pd.read_csv(adjectives_path)

    df['definitions'] = ['' for _ in range(len(df))]

    for index, row in tqdm(df.iterrows(), total=len(df)):

        prompt = f"Give me a short definition of the adjective {row['adjective']}. Answer in a sinle line with the format: {row['adjective']} is defined as ..."

        response = get_response(prompt)
        df.at[index, 'definitions'] = response

    df.to_csv(adjectives_path.replace(adjectives_path.split('/')[-1], 'adjectives_with_definitions.csv'), index=False)

if __name__ == "__main__":
    fire.Fire(main)