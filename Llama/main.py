import os
import sys
import traceback
import pickle
from pathlib import Path

from fire import Fire
import pandas as pd
import dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append('..')
dotenv.load_dotenv()

from inference import model_fn, predict_fn, get_probabilities_with_context, get_probabilities_no_context


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


def run_inference(
    input_files: str | list[str],
    adjectives_file: str,
    use_context: bool,
    repository: str,
    batch_size: int = 244
):
    """
    Unified inference runner for both context and no-context scenarios.
    
    Args:
        input_files: List of input CSV files to process or single file path
        adjectives_file: Path to the adjectives CSV file
        use_context: Whether to use context-based inference (default: False)
        repository: HuggingFace model repository
        batch_size: Batch size for inference
    """
    
    # Handle single file input
    if isinstance(input_files, str):
        input_files = [input_files]
    
    # Setup model
    model_id = repository.split("/")[-1]
    model_path = f'/workspace/{model_id}'
    
    # Clone model if it doesn't exist
    if not os.path.exists(model_path):
        clone_cmd = f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} {model_path}'
        print(f"Cloning model: {clone_cmd}")
        os.system(clone_cmd)
    
    # Load model and tokenizer
    print(f"{bcolors.OKBLUE}Loading model and tokenizer...{bcolors.ENDC}")
    model_and_tokenizer = model_fn(model_path, history_size=batch_size, use_context=use_context)

    if model_and_tokenizer is None:
        print(f"{bcolors.FAIL}Failed to load model!{bcolors.ENDC}")
        return
    
    # Load adjectives
    df_adj = pd.read_csv(adjectives_file)
    special_tokens = df_adj['adjective'].tolist()
    print(f"{bcolors.OKGREEN}Loaded {len(special_tokens)} adjectives{bcolors.ENDC}")
    
    # Define prompt formats
    if use_context:
        prompt_format = '{context}\n"A": {comment}\nDoes response message from "A" seem "{word}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    else:
        prompt_format = '"A": {sentence}\nDoes "A" seem "{word}". Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

    
    print(f"{bcolors.OKBLUE}Running test...{bcolors.ENDC}")
    if use_context:
        test_input = [
            prompt_format.format(word='racist', context='how are you', comment='I hate black people'),
            prompt_format.format(word='racist', context='how are you', comment='we are all the same')
        ]
    else:
        test_input = [
            prompt_format.format(word='racist', sentence='Ich hasse schwarze Menschen'),
            prompt_format.format(word='racist', sentence='Wir sind alle gleich')
        ]
    output = predict_fn(
        data={'input': test_input, 'use_cache': True},
        model_and_tokenizer=model_and_tokenizer
    )
    print(f"Test output: {output}")
        
    
    # Process each input file
    for file_path in input_files:
        print(f"{bcolors.HEADER}Processing file: {file_path}{bcolors.ENDC}")
        
        if not os.path.exists(file_path):
            print(f"{bcolors.FAIL}File not found: {file_path}{bcolors.ENDC}")
            continue
        
        distributions = []
        current_batch_size = batch_size
        
        # Load data
        df = pd.read_csv(file_path, sep=',')[:3]
        df = df.dropna(subset=['text'])
        
        if use_context:
            df['text+context'] = df['text'] + ' ' + df['context']
            df['slen'] = df['text+context'].map(lambda x: len(x))
            df = df.sort_values(by=['slen'], ascending=True)
            
            ids = df['id'].tolist()
            messages = df['text'].tolist()
            contexts = df['context'].tolist()
            
            data_iterator = enumerate(zip(messages, contexts))
            total = len(messages)
        else:
            # No-context processing
            df['slen'] = df['text'].map(lambda x: len(x))
            df = df.sort_values(by=['slen'], ascending=True)
            
            ids = df['id'].tolist()
            sentences = df['text'].tolist()
            sentences = [s for s in sentences if len(s) > 0]
            data_iterator = enumerate(sentences)
            total = len(sentences)
        
        print(f"{bcolors.OKGREEN}Running inference for {total} instances in {file_path}!{bcolors.ENDC}")
        
        # Process data
        for j, data_item in tqdm(data_iterator, total=total):
            
            while current_batch_size > 0:
                try:
                    if use_context:
                        msg, cont = data_item
                        probabilities = get_probabilities_with_context(
                            model_and_tokenizer=model_and_tokenizer,
                            special_tokens=special_tokens,
                            prompt_format=prompt_format,
                            comment=msg,
                            context=cont,
                            batch_size=current_batch_size
                        )
                    else:
                        sentence = data_item
                        probabilities = get_probabilities_no_context(
                            model_and_tokenizer=model_and_tokenizer,
                            special_tokens=special_tokens,
                            prompt_format=prompt_format,
                            sentence=sentence,
                            batch_size=current_batch_size
                        )
                    
                    distributions.append(probabilities)
                    break
                    
                except Exception as e:
                    print(traceback.format_exc())
                    current_batch_size -= 1
                    print(f"{bcolors.FAIL}Batch size {current_batch_size} failed!{bcolors.ENDC}")
                    if current_batch_size <= 0:
                        print(f"{bcolors.FAIL}All batch sizes failed for item {j}!{bcolors.ENDC}")
                        break
            
            # Save intermediate results every 100 items
            if j % 100 == 0 and j > 0:
                output_file = f'{file_path}.pickle'
                with open(output_file, 'wb') as handle:
                    save_data = {
                        'id': ids[:len(distributions)],
                        'values': distributions
                    }
                    if not use_context:
                        save_data['text'] = sentences[:len(distributions)]
                    
                    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"{bcolors.OKCYAN}Intermediate save at {j} items{bcolors.ENDC}")
        
        # Final save
        output_file = f'{file_path}.pickle'
        with open(output_file, 'wb') as handle:
            save_data = {
                'id': ids[:len(distributions)],
                'values': distributions
            }
            if not use_context:
                save_data['text'] = sentences[:len(distributions)]
            
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"{bcolors.OKGREEN}Completed processing {file_path}. Output saved to {output_file}{bcolors.ENDC}")


# Fire CLI entry point with explicit parameters
def main(
    input_files: str | list[str],
    use_context: bool,
    adjectives_file: str = "../AdjectiveSetGeneration/adjectives.csv",
    repository: str = "meta-llama/Llama-3.1-8B-Instruct",
    batch_size: int = 244
):
    """Main entry point using Fire for CLI. Explicit parameters for clarity."""
    return run_inference(
        input_files=input_files,
        adjectives_file=adjectives_file,
        use_context=use_context,
        repository=repository,
        batch_size=batch_size
    )


if __name__ == "__main__":
    Fire(main)
