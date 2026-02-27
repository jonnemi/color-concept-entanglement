# Probing Color-Concept Entanglement in Vision-Language Models

This master’s thesis project investigates how vision-language models (VLMs) attribute color to objects under controlled perceptual evidence.

To study this, I constructed the Graded Color Attribution (GCA) dataset: black-and-white outline drawings in which the proportion of foreground pixels assigned a target color is varied from 0% to 100%. This allows precise measurement of the threshold at which models (and humans) report that an object “is” a given color.

The project includes:

Dataset construction:
Objects from Visual CounterFact are retrieved as clean outline drawings via Google Custom Search. Images are filtered using GPT-based scoring and manual verification. Foreground masks are generated using OpenCV-based segmentation, and recoloring is applied only to white object pixels at controlled thresholds.

Model evaluation:
VLMs are prompted with “What color is the [OBJECT] in the image?” to measure color attribution behavior under prior, counterfactual, and control (shape) conditions. We additionally test whether models follow their own stated color-threshold rules.

Human study:
Participants complete the same task via a web interface (Flask), allowing direct human–model comparison of attribution thresholds and introspection consistency.
