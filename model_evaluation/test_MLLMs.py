from transformers import (
    BitsAndBytesConfig,LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from PIL import Image
import torch
import pandas as pd
import re
import argparse
from tqdm import tqdm
import os
import io
import gc
import base64
from pathlib import Path
import torch.nn.functional as F
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import asyncio

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()
client = OpenAI()
client_async = AsyncOpenAI()


def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def create_eval_prompt(object_name, most="False"):
    instruction_tokens = "[INST] <image>\n"
    end_tokens = "[/INST]"
    #question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
    if most == "True":
        object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
        question = f"What color are most {object_name_plural}?"

    else:
        question = f"What color is this {object_name}?"

    prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"

    return prompt


def prompt_mllm(df, processor, model, device, prompt, dummy=False, return_probs=False):
    """
    Run inference for a batch of images and optionally compute P(correct_answer)
    using the final layer logits.
    """
    with torch.inference_mode():
        torch.cuda.empty_cache()
        gc.collect()
        generated_texts = []
        probs_correct = []

        for idx, row in df.iterrows():
            if dummy:
                #dummy_image = Image.new("RGB", (256, 256), color="white")
                #image = None
                inputs = processor(text=prompt, return_tensors='pt')
            else:
                try:
                    image = Image.open(row['image_path']).convert("RGB")
                    #image = image.resize((256, 256), Image.LANCZOS)
                    inputs = processor(images=image, text=prompt, return_tensors='pt')
                except FileNotFoundError:
                    print(f"Warning: Image not found for {row['object']}")
                    generated_texts.append(None)
                    probs_correct.append(None)
                    continue  # Skip to the next row in the DataFrame
            
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Perform a forward pass with the model
            outputs = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
            predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = clean_instruction_tokens(predicted_answer)
            generated_texts.append(predicted_answer.lower().replace("gray", "grey"))

            if return_probs:
                with torch.inference_mode():
                    outputs_logits = model(**inputs)
                    logits = outputs_logits.logits[:, -1, :]  # final token logits
                    probs_softmax = F.softmax(logits, dim=-1).squeeze(0).detach().cpu()

                correct = str(row["correct_answer"]).strip().lower()
                correct_ids = processor.tokenizer(correct).input_ids
                correct_ids_cap = processor.tokenizer(correct.capitalize()).input_ids
                token_idx = 1 if "llava" in type(model).__name__.lower() else 0

                try:
                    prob_correct = max(
                        probs_softmax[correct_ids[token_idx]].item(),
                        probs_softmax[correct_ids_cap[token_idx]].item()
                    )
                except Exception:
                    prob_correct = None

                probs_correct.append(prob_correct)

            # cleanup
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

        df['predicted_color'] = generated_texts
        if return_probs:
            df['prob_correct'] = probs_correct

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return df


def encode_image_to_b64(path):
    """Return base64 string for a local image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    


def prompt_gpt(df, prompt, model_name="gpt-4o", dummy=False, return_probs=False):
    """
    GPT equivalent of prompt_mllm().
    Matches output format:
        df['predicted_color']
        df['prob_correct'] (always None, placeholder)
    """

    preds = []
    probs = []

    for _, row in df.iterrows():

        # Build input image
        if dummy:
            img = Image.new("RGB", (256, 256), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            try:
                img_b64 = encode_image_to_b64(row["image_path"])
            except FileNotFoundError:
                preds.append(None)
                probs.append(None)
                continue

        # GPT query
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                max_tokens=10,
                temperature=0.0,
                top_p=0,
            )
            ans = response.choices[0].message.content.strip().lower()
            ans = ans.replace("gray", "grey").split()[0]
        except Exception as e:
            print("GPT error:", e)
            ans = None

        preds.append(ans)
        probs.append(None)   # placeholder to match MLLM interface

    df["predicted_color"] = preds
    if return_probs:
        df["prob_correct"] = probs

    return df



async def prompt_gpt_async(df, prompt, model_name="gpt-4o", dummy=False):
    """
    Async GPT equivalent of prompt_gpt().
    Returns df with a new column: predicted_color.
    All requests are sent concurrently.
    """

    async def query_single(row):
        # Build / encode image
        if dummy:
            img = Image.new("RGB", (256, 256), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            try:
                img_b64 = encode_image_to_b64(row["image_path"])
            except FileNotFoundError:
                return None

        try:
            response = await client_async.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0.0,
                top_p=0,
            )

            ans = response.choices[0].message.content.strip().lower()
            return ans.split()[0].replace("gray", "grey")

        except Exception as e:
            print("GPT error:", e)
            return None

    # Build tasks (no batching)
    tasks = [query_single(row) for _, row in df.iterrows()]

    # Run in parallel
    preds = await asyncio.gather(*tasks)

    df = df.copy()
    df["predicted_color"] = preds
    return df


def run_async(coro):
    """
    Safe asyncio runner: 
    - uses asyncio.run() when no loop is running
    - uses await via nest_asyncio when in Jupyter
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside Jupyter
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        # Normal Python script
        return asyncio.run(coro)
    

def prompt_gpt_sync(df, prompt, model_name="gpt-4o", dummy=False):
    return run_async(prompt_gpt_async(df, prompt, model_name=model_name, dummy=dummy))


def run_vlm_evaluation(
    df,
    processor=None,
    model=None,
    device=None,
    batch_size=1,
    mode="this",  # "this", "most", or "both"
    dummy=False,
    return_probs=False,
    backend="llava",        # "llava" | "gpt" | "qwen"
    model_name=None,
):
    """
    Generic evaluation loop for Vision-Language Models.

    backend:
        - "llava": open-weight torch VLMs (LLaVA, etc.)
        - "gpt":   OpenAI GPT models (default: gpt-5o)
        - "qwen":  Qwen-VL models (HF)
    """

    assert mode in ["this", "most", "both"]
    assert backend in ["llava", "gpt", "qwen"]

    results = []

    # Select backend caller
    if backend == "gpt":
        gpt_model = model_name or "gpt-5o"

        def caller(batch, prompt):
            return prompt_gpt_sync(
                batch,
                prompt,
                model_name=gpt_model,
                dummy=dummy,
            )

    elif backend == "qwen":
        # Qwen behaves like other torch VLMs
        def caller(batch, prompt):
            return prompt_mllm(
                batch,
                processor=processor,
                model=model,
                device=device,
                prompt=prompt,
                dummy=dummy,
                return_probs=return_probs,
            )

    else:  # "llava"
        def caller(batch, prompt):
            return prompt_mllm(
                batch,
                processor=processor,
                model=model,
                device=device,
                prompt=prompt,
                dummy=dummy,
                return_probs=return_probs,
            )

    # Main loop
    for i in tqdm(
        range(0, len(df), batch_size),
        desc=f"Running VLM ({backend})",
        position=1,
        leave=False,
    ):
        batch_df = df.iloc[i : i + batch_size].copy()

        if mode in ["most", "both"]:
            prompt = create_eval_prompt(batch_df["object"], most="True")
            df_most = caller(batch_df, prompt)
            df_most = df_most.rename(columns={
                "predicted_color": "pred_color_most",
                "prob_correct": "prob_correct_most" if return_probs else None,
            })
        else:
            df_most = None

        if mode in ["this", "both"]:
            prompt = create_eval_prompt(batch_df["object"], most="False")
            df_this = caller(batch_df, prompt)
            df_this = df_this.rename(columns={
                "predicted_color": "pred_color_this",
                "prob_correct": "prob_correct_this" if return_probs else None,
            })
        else:
            df_this = None

        if mode == "both":
            result_df = pd.merge(
                df_most,
                df_this,
                on=["image_path", "object", "correct_answer"],
                how="inner",
            )
        elif mode == "most":
            result_df = df_most
        else:
            result_df = df_this

        results.append(result_df)

        # memory hygiene
        del result_df
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return pd.concat(results, ignore_index=True)



def run_vlm_evaluation_old(
    df,
    processor=None,
    model=None,
    device=None,
    batch_size=1,
    mode="both",  # "this", "most", or "both"
    dummy=False,
    return_probs=False,
    use_gpt=False,
):
    """
    Generic evaluation loop for Vision-Language Models.

    Runs mllm_testing over a dataset for either:
      - "this" question type (What color is this object?)
      - "most" question type (What color are most objects?)
      - or "both" (runs both and merges results)

    Args:
        df: DataFrame with at least ['image_path', 'object', 'image_type']
        processor: model processor (e.g., LlavaNextProcessor)
        model: VLM model
        batch_size: how many rows per iteration
        mode: "this", "most", or "both"
        dummy_image: pass-through flag for special testing
        return_probs: whether to compute P(correct_answer) from logits

    Returns:
        DataFrame with added predicted color columns
    """

    assert mode in ["this", "most", "both"], "mode must be one of ['this', 'most', 'both']"

    results = []

    # Choose GPT or open-weight MLLM caller
    if use_gpt:
        gpt_model_name = "gpt-4o"
        caller = lambda batch, prompt: prompt_gpt_sync(
            batch, prompt, model_name=gpt_model_name, dummy=dummy
        )

    else:
        caller = lambda batch, prompt: prompt_mllm(
            batch, processor, model, device,
            prompt=prompt, dummy=dummy, return_probs=return_probs
        )

    for i in tqdm(range(0, len(df), batch_size), desc=f"Running VLM ({'GPT' if use_gpt else 'Torch'})", position=1, leave=False):
        batch_df = df.iloc[i : i + batch_size].copy()

        with torch.inference_mode():
            if mode in ["most", "both"]:
                prompt = create_eval_prompt(batch_df["object"], most="True")
                df_most = caller(batch_df, prompt=prompt)
                df_most = df_most.rename(columns={
                    "predicted_color": "pred_color_most",
                    "prob_correct": "prob_correct_most" if return_probs else None
                })
            else:
                df_most = None

            if mode in ["this", "both"]:
                prompt = create_eval_prompt(batch_df["object"], most="False")
                df_this = caller(batch_df, prompt=prompt)
                df_this = df_this.rename(columns={
                    "predicted_color": "pred_color_this",
                    "prob_correct": "prob_correct_this" if return_probs else None
                })
            else:
                df_this = None
            
            
            # Merge results
            if mode == "both":
                result_df = pd.merge(
                    df_most,
                    df_this,
                    on=["image_path", "object", "image_variant", "correct_answer"],
                    how="inner"
                )
            elif mode == "most":
                result_df = df_most
            else:
                result_df = df_this

        results.append(result_df)

        # memory hygiene
        del result_df
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    df_results = pd.concat(results, ignore_index=True)
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next'], required=True, help="Choose the model version.")
    # NOTE: all images now have a perspective line. Keeping in to not mess up file names. 
    #parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=False, help="Choose dataset size (mini or full).")
    #parser.add_argument('--image_type', type=str, choices=['color', 'grayscale'], required=True, help="Choose color or grayscale image.")
    parser.add_argument("--mode", type=str, choices=["this", "most", "both"], default="both", help="Evaluation mode: 'this', 'most', or 'both'.")
    parser.add_argument("--dummy_image", action="store_true", help="Use a dummy white image instead of the real one (for model priors).", default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set a specific seed for reproducibility
    SEED = 42
    # Setting the seed for PyTorch
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if args.model_version == 'llava-next':
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", quantization_config=bnb_config
        ).to(device)
                
    project_root = Path.cwd()
    data_folder = project_root / "data" / "fruit"

    # Dataset path
    dataset_path = (
        data_folder / "fruit_images.parquet"
    )

    df = pd.read_parquet(dataset_path)
        
    print(f"Running evaluation: mode={args.mode}, dummy={args.dummy_image}")
    df_results = run_vlm_evaluation(
        df=df,
        processor=processor,
        model=model,
        device=device,
        batch_size=1,
        mode=args.mode,
        dummy=args.dummy_image,
    )
    
    out_name = (
        f"results_{args.model_version}_{args.mode}.csv"
    )
    out_path = data_folder / out_name

    df_results.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")

    
if __name__ == "__main__":
    main()