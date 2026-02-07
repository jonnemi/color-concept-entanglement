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


def create_eval_prompt(
    object_name: str,
    *,
    most: bool = False,
    calibration_value: int | None = None,
):
    """
    Builds either a normal or calibrated prompt.
    """

    intro = ""
    if calibration_value is not None:
        intro = (
            f"For any object, {calibration_value}% of its pixels should be colored for it to be "
            "considered that color."
            "Please keep this threshold in mind when answering the next question."
        )

    if most:
        obj = object_name if object_name.endswith("s") else object_name + "s"
        question = f"What color are most {obj}?"
    else:
        question = f"What color is this {object_name}?"

    prompt = (
        f"{intro}"
        f"Answer with one word. {question}"
    )
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
                #dummy_image = Image.new("RGB", (512, 512), color="white")
                #image = None
                inputs = processor(text=prompt, return_tensors='pt')
            else:
                try:
                    image = Image.open(row['image_path']).convert("RGB")
                    #image = image.resize((512, 512), Image.LANCZOS)
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
                    probs_softmax = max(
                        probs_softmax[correct_ids[token_idx]].item(),
                        probs_softmax[correct_ids_cap[token_idx]].item()
                    )
                except Exception:
                    probs_softmax = None

                probs_correct.append(probs_softmax)

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


def prompt_qwen(df, processor, model, device, prompt):
    preds = []

    for _, row in df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=image,
            text=chat_text,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        raw = processor.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).lower()

        preds.append(raw.replace("gray", "grey").split()[0])

    df = df.copy()
    df["pred_color_this"] = preds
    return df


def encode_image_to_b64(path):
    """Return base64 string for a local image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    


def prompt_gpt(df, prompt, model_name="gpt-4o", dummy=False, return_probs=False):
    """
    GPT equivalent of prompt_mllm().
    Matches output format:
        df['pred_color_this']
        df['prob_correct_this'] (always None, placeholder)
    """

    preds = []
    probs = []

    for _, row in df.iterrows():

        # Build input image
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
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

    df["pred_color_this"] = preds
    if return_probs:
        df["prob_correct_this"] = probs

    return df


def prompt_gpt52(
    df,
    prompt,
    model_name="gpt-5.2",
    dummy=False,
    top_k=5,
):
    preds = []
    logprob_preds = []
    logprob_corrects = []
    correct_in_topk = []

    for _, row in df.iterrows():

        # --- image handling ---
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            img_b64 = encode_image_to_b64(row["image_path"])

        # --- GPT-5.2 request ---
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{img_b64}",
                        },
                    ],
                }
            ],
            max_output_tokens=16,
            temperature=0.0,
            output=[
                {
                    "type": "output_text",
                    "logprobs": {
                        "top_k": top_k
                    },
                }
            ],
        )

        # --- extract output_text ---
        text_items = [
            item for item in response.output
            if item["type"] == "output_text"
        ]

        if not text_items:
            preds.append(None)
            logprob_preds.append(None)
            logprob_corrects.append(None)
            correct_in_topk.append(False)
            continue

        out = text_items[0]
        tokens = out.get("tokens", [])

        if not tokens:
            preds.append(None)
            logprob_preds.append(None)
            logprob_corrects.append(None)
            correct_in_topk.append(False)
            continue

        # --- predicted token ---
        pred_token = (
            tokens[0]["token"]
            .lower()
            .replace("gray", "grey")
        )
        preds.append(pred_token)

        logprob_preds.append(tokens[0]["logprob"])

        # --- correct color handling ---
        correct = str(row["correct_answer"]).lower()
        found = False
        lp_correct = None

        for cand in tokens[0].get("top_logprobs", []):
            tok = cand["token"].lower().replace("gray", "grey")
            if tok == correct:
                found = True
                lp_correct = cand["logprob"]
                break

        correct_in_topk.append(found)
        logprob_corrects.append(lp_correct)

    df = df.copy()
    df["predicted_color"] = preds
    df["logprob_pred_token"] = logprob_preds
    df["logprob_correct_token"] = logprob_corrects
    df["correct_in_top_k"] = correct_in_topk

    return df


async def prompt_gpt_async(df, prompt, model_name="gpt-4o", dummy=False):
    """
    Async GPT equivalent of prompt_gpt().
    Returns df with a new column: pred_color_this.
    All requests are sent concurrently.
    """

    async def query_single(row):
        # Build / encode image
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
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
    df["pred_color_this"] = preds
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
    *,
    backend: str,                    # "llava" | "qwen" | "gpt4" | "gpt52"
    processor=None,
    model=None,
    device=None,
    model_name=None,
    calibration_value: int | None = None,
    mode="this",
):
    """
    Generic evaluation loop for Vision-Language Models.

    backend:
        - "llava": open-weight torch VLMs (LLaVA, etc.)
        - "gpt":   OpenAI GPT models (default: gpt-5o)
        - "qwen":  Qwen-VL models (HF)
    """
     
    results = []

    for _, row in df.iterrows():
        prompt = create_eval_prompt(
            row["object"],
            most=(mode == "most"),
            calibration_value=calibration_value,
        )

        if backend == "llava":
            prompt = f"[INST] <image>\n{prompt}\n[/INST]"
            out = prompt_mllm(
                df, processor, model, device, prompt, return_probs=True
            )

        elif backend == "qwen":
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            out = prompt_qwen(
                df, processor, model, device, prompt
            )

        elif backend == "gpt4":
            out = prompt_gpt(
                df, prompt, model_name="gpt-4o"
            )

        elif backend == "gpt52":
            out = prompt_gpt52(
                df, prompt, model_name="gpt-5.2"
            )

        else:
            raise ValueError(backend)

        out["calibration"] = calibration_value
        results.append(out)

    return pd.concat(results, ignore_index=True)


# Helpers for introspection prompt
INTROSPECTION_PROMPT = """For any object, x% of its pixels should be colored for it to be considered that color.
For example, imagine an image of a banana, where only part of the banana in the image is colored yellow.
At what point would you personally say that the banana in the image is yellow?
What value should x% be?
Please only answer with a single number between 0 and 100."""



def parse_percentage(text: str | None) -> int | None:
    if not text:
        return None
    matches = re.findall(r"\b(\d{1,3})\b", text)
    if not matches:
        return None
    value = int(matches[-1])   # ← take LAST number
    return value if 0 <= value <= 100 else None



DUMMY_IMAGE = Image.new("RGB", (512, 512), "white")

def ask_vlm_introspection_threshold(
    *,
    backend: str,                  # "llava" | "qwen" | "gpt"
    processor=None,
    model=None,
    device=None,
    model_name: str | None = None,
) -> dict:

    if backend == "llava":
        prompt = f"[INST] <image>\n{INTROSPECTION_PROMPT}\n[/INST]"

        inputs = processor(
            #images=DUMMY_IMAGE,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        raw = processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print(raw)
        raw = clean_instruction_tokens(raw).strip().lower()

    elif backend == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INTROSPECTION_PROMPT},
                ],
            }
        ]

        # build the chat-formatted text 
        chat_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        # Stokenize / preprocess into tensors
        inputs = processor(
            #images=DUMMY_IMAGE,
            text=chat_text,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        raw = processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip().lower()


    elif backend == "gpt":
        gpt_model = model_name or "gpt-5.2"

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": INTROSPECTION_PROMPT,
                }
            ],
            temperature=0.0,
            max_completion_tokens=50,
        )
        raw = response.choices[0].message.content.strip().lower()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    threshold = parse_percentage(raw)

    return {
        "backend": backend,
        "model_name": model_name,
        "introspection_raw": raw,
        "introspection_threshold": threshold,
    }


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