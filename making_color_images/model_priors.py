"""
model_priors.py

Module for generating and analyzing model-based color priors for visual-language models (VLMs),
for open-weight models e.g. LLaVA, Qwen in ModelColorPriors class, and for GPT-Vision models in GPTColorPriors class.
"""

import gc
import ast
import torch
import pandas as pd
from tqdm import tqdm
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image
from test_MLLMs import prompt_mllm
from collections import Counter


class BaseColorPriors:
    """
    Abstract parent class for color prior extraction.
    Contains all shared logic and utilities.
    """

    def __init__(self, model_name, data_folder):
        self.model_name = model_name
        self.data_folder = data_folder

    #  Abstract methods to override
    def query_model_dummy(self, df, question):
        """Return a color string (text-only)."""
        raise NotImplementedError

    def query_model_image(self, df, question, image_path):
        """Return a color string (image + question)."""
        raise NotImplementedError
    

    def create_prior_prompt(self, object_name, most=True):
        """
        Build a prompt that asks for up to 3 likely colors,
        ordered from most to least likely, comma-separated.
        Uses correct LLaVA instruction-token style.
        """

        instruction_tokens = "[INST] <image>\n"
        end_tokens = "[/INST]"

        # Build question
        if most == "True":
            plural = object_name if object_name.endswith("s") else object_name + "s"
            question = (
                f"List up to three possible colors for most {plural}, "
                f"from most likely to least likely."
            )
        else:
            question = (
                f"List up to three possible colors for this {object_name}, "
                f"from most likely to least likely."
            )

        # Style constraint
        format_rule = (
            "Respond ONLY with english color words separated by commas. "
            "Do not use sentences."
        )

        # Final prompt
        prompt = (
            f"{instruction_tokens}"
            f"{question} {format_rule}"
            f" {end_tokens}"
        )

        return prompt
        

    def get_model_color_priors(self, df, most=True, save=True):
        """
        Shared implementation:
        Calls subclass methods for priors with and without image.
        """
        results = []
        batch_size = 1  # larger batch sizes not implemented yet

        def _parse_colors(s):
            """Convert 'red, green, blue' → ['red','green','blue']"""
            parts = [p.strip() for p in s.split(",") if p.strip()]
            
            return parts


        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Color priors ({self.model_name})"):
            batch_df = df.iloc[i : i + batch_size].reset_index(drop=True)
            object_name = row["object"]
            prompt = self.create_prior_prompt(object_name, most=str(most))

            dummy_priors = self.query_model_dummy(batch_df, prompt)
            img_priors  = self.query_model_image(batch_df, prompt)

            out_batch = {
                "object": row["object"],
                "correct_answer": row["correct_answer"],
                "dummy_priors": _parse_colors(dummy_priors[0]),
                "image_priors": _parse_colors(img_priors[0]),
            }

            results.append(out_batch)
            del out_batch
            del batch_df
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        ground_truth_df = pd.DataFrame(results)       

        if save:
            out_path = self.data_folder / f"color_priors_{self.model_name}.csv"
            ground_truth_df.to_csv(out_path, index=False)
            print("Saved priors to: ", out_path)

        return ground_truth_df

    def pick_primary_color(self, df, column="dummy_priors"):
        """
        Select the best (primary) color for each row:
        • keep only allowed colors
        • remove excluded colors
        • if nothing remains return NaN (None)
        • otherwise return first remaining color
        """

        EXCLUDE = {"white", "silver", "gold", "clear"}

        ALLOWED = {
            "red", "brown", "pink", "orange", "yellow", "gold",
            "green", "blue", "purple", "black", "grey", "silver", "white"
        }

        primary_colors = []

        for obj, priors in zip(df["object"], df[column]):
            # Ensure list
            if not isinstance(priors, list):
                print(f"[WARN] priors for {obj} not a list: {priors}")
                primary_colors.append(None)
                continue

            # Normalize input
            priors = [str(c).lower().strip() for c in priors]

            # Keep only valid allowed colors
            filtered = [c for c in priors if c in ALLOWED and c not in EXCLUDE]

            if len(filtered) == 0:
                # nothing valid left
                print(f"[NULL] {obj}: all priors invalid ({priors}) writing NaN")
                primary_colors.append(None)  # → becomes NaN in DataFrame
                continue

            # If something remains, pick the first valid one
            chosen = filtered[0]

            # Print when a correction occurred
            if chosen != priors[0]:
                print(f"[INFO] {obj}: replaced '{priors[0]}' with '{chosen}'")

            primary_colors.append(chosen)

        return primary_colors


    # Shared analysis
    def analyze_differences(self, df):
        """
        Add diagnostic columns for dummy vs image priors.
        """
        df = df.copy()

        # Check if model_prior color in GT
        df["prior_in_gt"] = df.apply(
            lambda r: r["prior"].lower()
            in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )
        print(f"{df['prior_in_gt'].sum()} rows where the chosen model color prior is NOT in ground truth from Visual Counterfact.")

        model_priors = df["prior"].unique()
        print(f"Model color priors: {model_priors}")


    def load_model_priors(self):
        path = self.data_folder / f"color_priors_{self.model_name}.csv"
        df = pd.read_csv(path)
        df["correct_answer"] = df["correct_answer"].apply(ast.literal_eval)
        df["dummy_priors"] = df["dummy_priors"].apply(ast.literal_eval)
        df["image_priors"] = df["image_priors"].apply(ast.literal_eval)
        return df


class TorchColorPriors(BaseColorPriors):
    """
    Color prior extraction using open-weight VLMs (with processor and torch model).
    """
    def __init__(self, processor, model, device, data_folder):
        super().__init__(model.name_or_path.split("/")[-1], data_folder)
        self.processor = processor
        self.model = model
        self.device = device

    def query_model_dummy(self, df, prompt):
        result = prompt_mllm(
            df,
            self.processor,
            self.model,
            self.device,
            prompt=prompt,
            dummy=True
        )
        return result["predicted_color"].tolist()

    def query_model_image(self, df, prompt):
        result = prompt_mllm(
            df,
            self.processor,
            self.model,
            self.device,
            prompt=prompt,
            dummy=False
        )
        return result["predicted_color"].tolist()


"""
Color-prior extraction using GPT-Vision models (no processor, no torch model).
"""
client = OpenAI()

def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class GPTColorPriors:
    def __init__(self, model_name: str, data_folder: Path):
        """
        Args:
            model_name  (str): e.g., "gpt-4o", "gpt-4.1", "gpt-4o-mini"
            data_folder (Path): where priors are saved
        """
        self.model_name = model_name
        self.data_folder = data_folder


    # Core query helpers
    def ask_gpt(self, question: str, image_b64: str | None = None):
        """
        Send a text-only or image+text message to GPT-Vision.
        Always returns a LOWERCASE single-word predicted color.
        """
        if image_b64 is None:
            # TEXT-ONLY PRIOR
            messages = [
                {"role": "user", "content": question}
            ]
        else:
            # IMAGE + QUESTION
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ]

        response = client.responses.create(
            model=self.model_name,
            input=messages
        )

        out = response.output_text.strip().lower()
        out = out.split()[0]         # take the first word
        out = out.replace("gray", "grey")
        return out

    # -------------------------------------------------------------
    # MAIN: Get model color priors
    # -------------------------------------------------------------
    def get_model_color_priors(self, df, most="True", save=True):
        """
        Compute two priors:
           → model_prior_dummy : text-only
           → model_prior_real  : grayscale + prompt

        Args:
            df: DataFrame with column "object" and "image_path"
            most: "True" → use plural ("most apples")
            save: save a CSV file

        Returns: DataFrame with priors added
        """
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="GPT Color Priors"):

            object_name = row["object"]
            if most == "True":
                plural = object_name if object_name.endswith("s") else object_name + "s"
                question = f"Answer with one word. What color are most {plural}?"
            else:
                question = f"Answer with one word. What color is this {object_name}?"

            # ---- Dummy (true prior: NO IMAGE) ----
            prior_dummy = self.ask_gpt(question, image_b64=None)

            # ---- Real: grayscale image ----
            img = Image.open(row["image_path"]).convert("L")
            img = img.resize((256, 256))
            tmp_path = "/tmp/_gray.png"
            img.save(tmp_path)
            img_b64 = encode_image_base64(tmp_path)

            prior_real = self.ask_gpt(question, image_b64=img_b64)

            out_row = {
                "object": row["object"],
                "correct_answer": row["correct_answer"],
                "model_prior_dummy": prior_dummy,
                "model_prior": prior_real
            }
            results.append(out_row)

            gc.collect()

        priors_df = pd.DataFrame(results)

        # Save
        if save:
            out_path = self.data_folder / f"color_priors_{self.model_name}.csv"
            priors_df.to_csv(out_path, index=False)
            print(f"Saved GPT color priors → {out_path}")

        return priors_df

    # -------------------------------------------------------------
    def analyze_differences(self, priors_df):
        """
        Same logic as your old analyzer.
        """
        df = priors_df.copy()

        df["diff_dummy_vs_real"] = (
            df["model_prior_dummy"].str.lower()
            != df["model_prior"].str.lower()
        )

        print(f"{df['diff_dummy_vs_real'].sum()} rows differ between dummy and real priors.")

        # convert correct_answer to list
        df["correct_answer"] = df["correct_answer"].apply(
            lambda x: x if isinstance(x, list) else ast.literal_eval(x)
        )

        df["color_in_gt"] = df.apply(
            lambda r: r["model_prior"] in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )

        print(f"{(~df['color_in_gt']).sum()} priors NOT in GT.")

        print("Unique priors:", df["model_prior"].unique())

        return df

    
    def replace_correct_answers(self, df, priors_df, exclude=None, prior_col="model_prior"):
        """
        Replace df['correct_answer'] with model priors.
        """
        if exclude is None:
            exclude = ["white", "gold", "silver", "clear"]

        use_df = priors_df[~priors_df[prior_col].isin(exclude)]

        df = df.merge(
            use_df[["object", prior_col]],
            on="object",
            how="inner"
        )
        df["correct_answer"] = df[prior_col]
        df = df.drop(columns=[prior_col])

        print("Updated DF size:", df.shape[0])
        return df

    
    def load_model_priors(self):
        path = self.data_folder / f"color_priors_{self.model_name}.csv"
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} priors from {path}")
        return df

