# Color-Concept Entanglement: Investigating Color Attribution Thresholds in Humans and Vision-Language Models

This repository contains the code and stimulus generation pipeline for my master's thesis, which investigates how humans and vision-language models (VLMs) attribute color to partially colored objects, and whether the reasoning statements VLMs produce faithfully reflect their actual decision behavior.

## Dataset

The **Graded Color Attribution (GCA) dataset** is publicly available on HuggingFace:
👉 [mgolov/graded-color-attribution](https://huggingface.co/datasets/mgolov/graded-color-attribution)

The dataset consists of black-and-white outline drawings in which the proportion of foreground pixels assigned a target color (τ) is systematically varied from 0% to 100% across 13 threshold levels. This allows precise measurement of the threshold at which models and humans report that an object "is" a given color. Three stimulus conditions are included: canonical color priors, counterfactual color priors, and geometric shapes (perceptual baseline).

## Project Overview

### Dataset Construction
Object categories from the [Visual CounterFact (VCF)](https://arxiv.org/abs/2505.17127) dataset are used as a starting point. Clean black-and-white outline drawings are retrieved via Google Custom Search, filtered using GPT-4o-based scoring and manual verification, and segmented using an OpenCV-based pipeline. Recoloring is applied patch-wise (16×16px) to foreground pixels at controlled threshold levels using HSV manipulation.

### Model Evaluation
Three VLMs are evaluated on the GCA stimuli: GPT-4o, LLaVA-NeXT, and Qwen3-VL. Models are prompted to identify the color of a partially colored object under canonical, counterfactual, and shape conditions. We additionally test whether models follow their own stated color-threshold rules by injecting the model's self-reported threshold back into the prompt (calibrated evaluation).

### Human Study
173 participants completed the same color attribution task via a custom web interface (Flask + Supabase). This allows direct human–model comparison of attribution thresholds, certainty patterns, and introspection consistency.


## Repository Structure
├── data/                  # Stimulus images - now publicly available on HuggingFace (see above)
├── making_color_images/   # Full pipeline for stimulus generation: Google Custom Search retrieval,
│                          # GPT-4o scoring, OpenCV segmentation, and HSV-based patch-wise coloring
├── making_fruit_images/   # Early pilot experiments (not part of the final GCA dataset)
├── model_evaluation/      # VLM evaluation scripts, prompt templates, and result plotting
└── prolific_study/        # Human study web application (Flask + Supabase), survey profiles,
                           # and analysis scripts for human behavioral results


## Thesis
Master Thesis in Machine Learning, University of Tübingen, 2026.
Supervisors: Dr. Michal Golovanevsky (Brown University) · Dr. William Rudman (University of Texas at Austin)
