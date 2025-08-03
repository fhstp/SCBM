from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import LlamaForCausalLM

import torch
import traceback
import pickle, os

def model_fn(model_dir, history_size):
    

    SYSTEM_PERSONA = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
    
    prefix = f'Consider what a person "A" states when racting to a "CONTEXT":\n\n"CONTEXT": '
    print('Inference with context!!!!!!!!!')
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model = LlamaForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config, device_map = 'cuda').requires_grad_(False) 
        model.resize_token_embeddings(len(tokenizer))
        print('Parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
      
        model.eval()

        if os.path.isfile(os.path.join(model_dir, 'history.pickle')):
            with open(os.path.join(model_dir, 'history.pickle'), 'rb') as handle:
                history = pickle.load(handle)
        else:
            prefix_tokenized = tokenizer(prefix, return_tensors='pt')
            prefix_key_values = list(model(**prefix_tokenized, use_cache = True).past_key_values)
            prefix_key_values = [(z[0].repeat(history_size, 1, 1, 1), z[1].repeat(history_size, 1, 1, 1)) for z in prefix_key_values]
            prefix_attention_mask = prefix_tokenized.attention_mask.repeat(history_size, 1)
    
            history = {'attention_mask': prefix_attention_mask, 'past_key_values': prefix_key_values}
    except:
        print(traceback.format_exc())
        return
    return model, tokenizer, history


def predict_fn(data, model_and_tokenizer):
    
    model, tokenizer, history = model_and_tokenizer
    
    input = data.pop("inputs", data)
    
    instances = data.get('input')
    use_cache = data.get("use_cache", False)

    with torch.no_grad():
        
        tokenized = tokenizer(instances, return_tensors="pt", padding = True)

        if tokenized.input_ids.size(0) != history['attention_mask'].size(0):
            truncated_history = [(z[0][:tokenized.input_ids.size(0)], z[1][:tokenized.input_ids.size(0)]) for z in history['past_key_values']]
            trucnated_attention_mask = history['attention_mask'][:tokenized.input_ids.size(0)]
        else:
            truncated_history = history['past_key_values']
            trucnated_attention_mask = history['attention_mask']

        
        if use_cache:
            model_output = model(input_ids = tokenized.input_ids, 
                               attention_mask = torch.cat((trucnated_attention_mask, tokenized.attention_mask), dim=-1) ,
                               use_cache=use_cache, 
                               past_key_values=truncated_history)
        else:
            model_output = model(input_ids = tokenized.input_ids, 
                               attention_mask = tokenized.attention_mask)        
    
        logits = model_output.logits.detach().cpu()
        
        #take non-masked logits tokens
        last_nonzero_idx = torch.argmax(tokenized['attention_mask'].cumsum(1), dim=1)
        
        # FIX THIS COCHINADA
        logits = torch.stack([logits[i][non_zero] for i, non_zero in enumerate(last_nonzero_idx.tolist())])
        
        #! Here we apply softmax to the logits to estimate the probability of predicting yes tokens
        probs = torch.nn.functional.softmax(logits, dim=-1)
        #! Since the yes token may have several variants, e.g., "YES", "yes", "Yes\n", we sum the probabilities of all these tokens.
        #! The indices of the yes tokens are hardcoded, as they are known in advance, you may find them in the tokenizer file in the 
        #! HuggingFace repository of the model.
        probs = torch.sum(probs[:, [9642, 9891, 14331, 20137, 41898, 58841, 60844, 77830, 95934, 85502, 98171 , 5697, 45280, 53545]], dim=-1)
        
    return {"probs": probs}

def get_probabilities_withcontext(model_and_tokenizer, special_tokens, prompt_format, comment, context, batch_size):
    sorted_instances = [(prompt_format.format(word=x,
                                              comment=comment,
                                              context = context), idx) for idx, x in enumerate(special_tokens)]
    
    sorted_instances = sorted(sorted_instances, key = lambda x: -len(x[0]))    
    
    probabilities = None
    
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader([i[0] for i in sorted_instances], batch_size=batch_size, shuffle=False )
        for batch in dataloader:
            output = predict_fn({'input' : batch, 'use_cache' : True},
                               model_and_tokenizer = model_and_tokenizer)
            probs = output['probs'].clone().detach()
            probabilities = torch.cat([probabilities, probs]) if probabilities is not None else probs
        del dataloader

    values = sorted([(x.item(), y[-1]) for x, y in zip(probabilities, sorted_instances)], key = lambda x: x[1])
    return [x[0] for x in values]
