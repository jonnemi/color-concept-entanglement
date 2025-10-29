from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,
    LlavaForConditionalGeneration, LlavaNextProcessor,GenerationConfig,
    LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration,AutoModel, AutoTokenizer
)
from PIL import Image
import torch
import pandas as pd
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score

from PIL import Image
import matplotlib.pyplot as plt
import random

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

import base64

import argparse
import os
import pickle
import gc

from collections import defaultdict

from datasets import load_dataset
import io

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def mllm_testing(df, processor, model, model_name, task, image_type, most="True"):
    with torch.inference_mode():
        torch.cuda.empty_cache()
        gc.collect()
        generated_texts = []
        for idx, row in df.iterrows():
            instruction_tokens = "[INST] <image>\n"
            end_tokens = "[/INST]"
           
            if image_type == 'color':
                image_path = row['color_image']['bytes']
            else:
                image_path = row['grayscale_image']['bytes']
                
            object_name = row['object']
            #question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
            if most == "True":
                object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
                question = f"What color are most {object_name_plural}?"
                
            else:
                question = f"What color is this {object_name}?"
                
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"    
            
            try:
                image = Image.open(io.BytesIO(image_path)).convert("RGB")
                image = image.resize((256, 200) if task == "size" else (256, 256), Image.LANCZOS)
            except FileNotFoundError:
                print(f"Warning: Image not found for {row['object']}")
                generated_texts.append(None)
                continue  # Skip to the next row in the DataFrame
            
            inputs = processor(images=image, text=prompt, return_tensors='pt')
            inputs = {k: v.to('cuda') for k, v in inputs.items()} 
            # Perform a forward pass with the model
            outputs = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0)  # Adjust max_new_tokens as needed
            predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = clean_instruction_tokens(predicted_answer)
            
            generated_texts.append(predicted_answer)
            #print(torch.cuda.memory_summary())
            
            to_delete = ['inputs', 'outputs', 'image_inputs', 'video_inputs', 'generated_ids', 'prepare_inputs', 'image', 'pil_images', 'inputs_embeds']
            for var_name in to_delete:
                if var_name in locals():
                    var = locals()[var_name]
                    if isinstance(var, dict):
                        for v in var.values():
                            if torch.is_tensor(v) and v.is_cuda:
                                del v
                    elif torch.is_tensor(var) and var.is_cuda:
                        del var
                    del locals()[var_name]
            if hasattr(model, 'clear_kv_cache'):
                model.clear_kv_cache()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
            #print(torch.cuda.memory_summary())
                
    
        df['generated_text'] = generated_texts
    
        if 'inputs' in locals(): del inputs
        if 'image' in locals(): del image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return df

def main():
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next'], required=True, help="Choose the model version.")
    # NOTE: all images now have a perspective line. Keeping in to not mess up file names. 
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=False, help="Choose dataset size (mini or full).")
    parser.add_argument('--image_type', type=str, choices=['color', 'grayscale'], required=True, help="Choose color or grayscale image.")
    parser.add_argument('--most', type=str, choices=['True', 'False'], required=True, help="Choose if using 'this' or 'most'.")

    args = parser.parse_args()
    random.seed(0)
    print(torch.cuda.is_available())
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if args.model_version == 'llava-next':
        #processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb_config, device_map='auto')
        processor = LlavaNextProcessor.from_pretrained("unsloth/llava-1.5-7b-hf-bnb-4bit", quantization_config=bnb_config, device_map='auto')
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb_config, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        
    dataset = pd.read_parquet("data/fruit/fruit_images.parquet")
    #dataset = load_dataset("mgolov/Visual-Counterfact")

        
    #df = mllm_testing(df, processor, model, args.model_version, args.task, args.image_type, most=args.most)
    batch_size = 1
    results = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size].copy()
        with torch.inference_mode():
            result_df = mllm_testing(batch_df, processor, model, args.model_version, args.image_type, most=args.most)
            
        results.append(result_df)
        del result_df
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    
    df = pd.concat(results, ignore_index=True)

    df.to_csv(f'most_instances_plural_bigger_{args.task}_new_MLLM_results_most_{args.most}_{args.image_type}_line_{args.line}_{args.model_version}_{args.dataset_size}.csv', index=False)

    
if __name__ == "__main__":
    main()