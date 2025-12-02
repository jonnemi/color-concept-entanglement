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
from test_MLLMs import mllm_testing


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


    #  Shared logic for computing priors
    def make_question(self, object_name, most=True):
        if most:
            plural = object_name if object_name.endswith("s") else object_name + "s"
            return f"Answer with one word. What color are most {plural}?"
        else:
            return f"Answer with one word. What color is this {object_name}?"
        

    def get_model_color_priors(self, df, most=True, save=True, batch_size = 1):
        """
        Shared implementation:
        Calls subclass methods for priors with and without image.
        """
        results = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Color priors ({self.model_name})"):
            batch_df = df.iloc[i:i+ batch_size].copy()
            object_name = row["object"]
            image_path = row["image_path"]
            question = self.make_question(object_name, most=most)

            prior_dummy = self.query_model_dummy(batch_df, question)
            prior_image = self.query_model_image(batch_df, question)

            df_merged = pd.concat([prior_dummy, prior_image[["model_prior_image"]]], axis=1)

            results.append(df_merged)
            del df_merged
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        ground_truth_df = pd.concat(results, ignore_index=True)       

        if save:
            out_path = self.data_folder / f"color_priors_{self.model_name}.csv"
            ground_truth_df.to_csv(out_path, index=False)
            print("Saved priors to: ", out_path)

        return ground_truth_df

    
    # Shared analysis
    def analyze_differences(self, df):
        """
        Add diagnostic columns for dummy vs image priors.
        """
        df = df.copy()

        df["diff_dummy_vs_image"] = (
            df["model_prior_dummy"].str.lower() != df["model_prior_image"].str.lower()
        )

        print(df["diff_dummy_vs_image"].sum(), " mismatches when querying color priors with vs without image.")

        # Normalize correct_answer into list
        df["correct_answer"] = df["correct_answer"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Check if model_prior color in GT
        df["prior_dummy_in_gt"] = df.apply(
            lambda r: r["model_prior_dummy"].lower()
            in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )

        df["prior_image_in_gt"] = df.apply(
            lambda r: r["model_prior_image"].lower()
            in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )

        print(f"{df['prior_dummy_in_gt'].sum()} rows where model color prior WITHOUT image NOT in ground truth.")
        print(f"{df['prior_image_in_gt'].sum()} rows where model color prior WITH image NOT in ground truth.")

        model_priors = df["model_prior_dummy"].unique()
        print(f"Model color priors: {model_priors}")

        return df

    def replace_correct_answers(self, df, ground_truth_df, colors_to_exclude=None, prior_col="model_prior_dummy"):
        """
        Replace df['correct_answer'] with the model's prior, filtering excluded colors.
        """

        if colors_to_exclude is None:
            colors_to_exclude = ["silver", "gold", "white", "clear"]

        print(f"Excluding colors: {colors_to_exclude}")
        ground_truth_df = ground_truth_df[
            ~ground_truth_df[prior_col].isin(colors_to_exclude)
        ]

        # Merge priors into df
        df = df.merge(
            ground_truth_df[["object", prior_col]],
            on="object",
            how="inner",
        )
        df["correct_answer"] = df[prior_col]
        df = df.drop(columns=[prior_col])

        print(f"Updated dataset now has {df.shape[0]} rows.")
        return df  


    def load_model_priors(self):
        path = self.data_folder / f"color_priors_{self.model_name}.csv"
        return pd.read_csv(path)


class TorchColorPriors(BaseColorPriors):
    """
    Color prior extraction using open-weight VLMs (with processor and torch model).
    """
    def __init__(self, processor, model, device, data_folder):
        super().__init__(model.name_or_path.split("/")[-1], data_folder)
        self.processor = processor
        self.model = model
        self.device = device

    def query_model_dummy(self, df, question):
        result = mllm_testing(
            df,
            self.processor,
            self.model,
            self.device,
            most=question,
            dummy=True
        )
        result = result.rename(columns={"predicted_color": "model_prior_dummy"})
        return result

    def query_model_image(self, df, question):
        result = mllm_testing(
            df,
            self.processor,
            self.model,
            self.device,
            most=question,
            dummy=False
        )
        result = result.rename(columns={"predicted_color": "model_prior_image"})
        return result


class ModelColorPriors:
    def __init__(self, processor, model, device, data_folder):
        """
        Initialize the ModelColorPriors utility.

        Args:
            processor: The VLM processor (e.g., LlavaNextProcessor)
            model: The VLM model (e.g., LlavaNextForConditionalGeneration)
            data_folder: Path object where results will be saved
            device: The device the model runs on ("cuda" or "cpu")
        """
        self.processor = processor
        self.model = model
        self.data_folder = data_folder
        self.device = device
        self.model_name = model.name_or_path.split("/")[-1]

    def get_model_color_priors(self, df, most="True", save=True):
        """
        Compare color priors using dummy (white) vs real images.

        Args:
            df: DataFrame with at least 'object', 'image_path', 'correct_answer'
            most: 'True' for plural question ("most apples"), 'False' for singular
            save: whether to save results to CSV

        Returns:
            ground_truth_df: DataFrame with model_prior and model_prior_dummy
        """
        batch_size = 1
        results = []

        for i in tqdm(range(0, len(df), batch_size), desc="Running model color priors"):
            batch_df = df.iloc[i:i + batch_size].copy()

            with torch.inference_mode():
                # Dummy white image
                df_dummy = mllm_testing(
                    batch_df, self.processor, self.model, self.device, most=most, dummy=True
                )
                df_dummy = df_dummy.rename(columns={"predicted_color": "model_prior_dummy"})

                # Real grayscale image
                df_real = mllm_testing(
                    batch_df, self.processor, self.model, self.device, most=most, dummy=False
                )
                df_real = df_real.rename(columns={"predicted_color": "model_prior"})

                df_merged = pd.concat([df_dummy, df_real[["model_prior"]]], axis=1)

            results.append(df_merged)
            del df_merged
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        ground_truth_df = pd.concat(results, ignore_index=True)
        
        name = f"color_priors_{self.model_name}.csv"
        if save:
            out_path = self.data_folder / name
            ground_truth_df.to_csv(out_path, index=False)
            print(f"Saved model priors to {out_path}")

        display_cols = ["object", "correct_answer", "model_prior_dummy", "model_prior"]
        return ground_truth_df


    def analyze_differences(self, ground_truth_df):
        """
        Add diagnostic columns for dummy vs real priors and correctness checks.
        """
        ground_truth_df = ground_truth_df.copy()

        # Dummy vs real difference
        ground_truth_df["diff_dummy_vs_real"] = (
            ground_truth_df["model_prior_dummy"].str.lower()
            != ground_truth_df["model_prior"].str.lower()
        )
        n_diff = ground_truth_df["diff_dummy_vs_real"].sum()
        print(f"{n_diff} rows differ between dummy and real priors.")

        # Normalize correct_answer into list
        ground_truth_df["correct_answer"] = ground_truth_df["correct_answer"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Check if model_prior color in GT
        ground_truth_df["color_in_gt"] = ground_truth_df.apply(
            lambda r: r["model_prior"].lower()
            in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )

        n_not_in_gt = (~ground_truth_df["color_in_gt"]).sum()
        print(f"{n_not_in_gt} rows where model color prior NOT in ground truth.")

        model_priors = ground_truth_df["model_prior"].unique()
        print(f"Model color priors: {model_priors}")

        return ground_truth_df

  
    def replace_correct_answers(self, df, ground_truth_df, colors_to_exclude=None, prior_col="model_prior"):
        """
        Replace df['correct_answer'] with the model's prior, filtering excluded colors.
        """

        if colors_to_exclude is None:
            colors_to_exclude = ["silver", "gold", "white", "clear"]

        print(f"Excluding colors: {colors_to_exclude}")
        ground_truth_df = ground_truth_df[
            ~ground_truth_df[prior_col].isin(colors_to_exclude)
        ]

        # Merge priors into df
        df = df.merge(
            ground_truth_df[["object", prior_col]],
            on="object",
            how="inner",
        )
        df["correct_answer"] = df[prior_col]
        df = df.drop(columns=[prior_col])

        print(f"Updated dataset now has {df.shape[0]} rows.")
        return df


    def load_model_priors(self):
        """
        Load model priors CSV and return a DataFrame.
        """
        path = self.data_folder / f"color_priors_{self.model_name}.csv"
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    


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

