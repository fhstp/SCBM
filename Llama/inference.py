from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import LlamaForCausalLM, DynamicCache

import torch
import traceback
import pickle
import os


def model_fn(model_dir, history_size, use_context=False):
    """
    Unified model function that handles both context and no-context scenarios.
    
    Args:
        model_dir: Path to the model directory
        history_size: Size of the history buffer
        use_context: Whether to use context in the prompts
    """
    
    system_persona = "You are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers."
    system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_persona}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
    
    if use_context:
        prefix = f'Consider what a person "A" states when racting to a "CONTEXT":\n\n"CONTEXT": '
        print('Inference with context!!!!!!!!!')
    else:
        prefix = f'Consider what a person "A" states: '
        print('Inference with no context!!!!!!!!!')
    
    print(f"System prompt: {system_prompt}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model = LlamaForCausalLM.from_pretrained(
            model_dir, 
            quantization_config=bnb_config, 
            device_map='cuda'
        ).requires_grad_(False)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # Create prefix with system prompt for caching
        full_prefix = system_prompt + prefix
        prefix_tokenized = tokenizer(full_prefix, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            prefix_key_values = list(model(**prefix_tokenized, use_cache=True).past_key_values)
        
        prefix_key_values = [(z[0].repeat(history_size, 1, 1, 1), z[1].repeat(history_size, 1, 1, 1)) 
                           for z in prefix_key_values]
        prefix_attention_mask = prefix_tokenized.attention_mask.repeat(history_size, 1)

        history = {'attention_mask': prefix_attention_mask, 'past_key_values': prefix_key_values}
        
    except Exception as e:
        print(f"Error loading model: {traceback.format_exc()}")
        return None
    
    return model, tokenizer, history


def predict_fn(data, model_and_tokenizer):
    """
    Unified prediction function for both context and no-context scenarios.
    """
    model, tokenizer, history = model_and_tokenizer
    
    instances = data.get('input')
    use_cache = data.get("use_cache", False)

    with torch.no_grad():
        tokenized = tokenizer(instances, return_tensors="pt", padding=True).to('cuda')

        if tokenized.input_ids.size(0) != history['attention_mask'].size(0):
            truncated_history = [(z[0][:tokenized.input_ids.size(0)], z[1][:tokenized.input_ids.size(0)]) 
                               for z in history['past_key_values']]
            truncated_attention_mask = history['attention_mask'][:tokenized.input_ids.size(0)]
        else:
            truncated_history = history['past_key_values']
            truncated_attention_mask = history['attention_mask']
        
        cache = DynamicCache.from_legacy_cache(truncated_history)
        
        if use_cache:
            model_output = model(
                input_ids=tokenized.input_ids,
                attention_mask=torch.cat((truncated_attention_mask, tokenized.attention_mask), dim=-1),
                use_cache=use_cache,
                past_key_values=cache
            )
        else:
            model_output = model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask
            )

        logits = model_output.logits.detach().cpu()
        
        # Take non-masked logits tokens
        last_nonzero_idx = torch.argmax(tokenized['attention_mask'].cumsum(1), dim=1)
        
        # Extract the last non-padded token logits for each sequence
        logits = torch.stack([logits[i][non_zero] for i, non_zero in enumerate(last_nonzero_idx.tolist())])
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sum probabilities for "yes" tokens (hardcoded indices for Llama model)
        yes_token_indices = [9642, 9891, 14331, 20137, 41898, 58841, 60844, 77830, 95934, 85502, 98171, 5697, 45280, 53545]
        probs = torch.sum(probs[:, yes_token_indices], dim=-1)
        
    return {"probs": probs}


def get_probabilities_with_context(model_and_tokenizer, special_tokens, prompt_format, comment, context, batch_size):
    """
    Get probabilities for context-based prompts.
    """
    sorted_instances = [
        (prompt_format.format(word=x, comment=comment, context=context), idx) 
        for idx, x in enumerate(special_tokens)
    ]
    
    sorted_instances = sorted(sorted_instances, key=lambda x: -len(x[0]))
    
    probabilities = None
    
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader([i[0] for i in sorted_instances], batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            output = predict_fn({'input': batch, 'use_cache': True}, model_and_tokenizer=model_and_tokenizer)
            probs = output['probs'].clone().detach()
            probabilities = torch.cat([probabilities, probs]) if probabilities is not None else probs
        del dataloader

    values = sorted([(x.item(), y[-1]) for x, y in zip(probabilities, sorted_instances)], key=lambda x: x[1])
    return [x[0] for x in values]


def get_probabilities_no_context(model_and_tokenizer, special_tokens, prompt_format, sentence, batch_size):
    """
    Get probabilities for no-context prompts.
    """
    
    sorted_instances = [
        (prompt_format.format(word=x, sentence=sentence), idx) 
        for idx, x in enumerate(special_tokens)
    ]

    sorted_instances = sorted(sorted_instances, key=lambda x: -len(x[0]))
    probabilities = None
    
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader([i[0] for i in sorted_instances], batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            output = predict_fn({'input': batch, 'use_cache': True}, model_and_tokenizer=model_and_tokenizer)
            probs = output['probs'].clone().detach()
            probabilities = torch.cat([probabilities, probs]) if probabilities is not None else probs
        del dataloader

    values = sorted([(x.item(), y[-1]) for x, y in zip(probabilities, sorted_instances)], key=lambda x: x[1])
    return [x[0] for x in values]
