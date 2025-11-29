"""
model_priors.py

Module for generating and analyzing model-based color priors for visual-language models (VLMs),
e.g., LLaVA, Qwen, etc.
"""

import gc
import ast
import torch
import pandas as pd
from tqdm import tqdm
from test_MLLMs import mllm_testing


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

        if save:
            out_path = self.data_folder / "model_color_priors.csv"
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


    def load_model_priors(self, filename="model_color_priors.csv"):
        """
        Load model priors CSV and return a DataFrame.
        """
        path = self.data_folder / filename
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df
