# Deep Learning Nanodegree Projects

Four hands-on projects covering neural networks for tabular data, computer
vision, information retrieval, and conditional image generation. Each project
contains its own notebooks, code, and setup instructions.

## Projects at a glance

| Project | Task | Main techniques | Start here |
|---|---|---|---|
| Diabetes risk prediction | Classify diabetes risk from health indicators | PyTorch MLP, preprocessing, threshold selection | [Project README](Diabetes%20Risk%20Prediction%20with%20PyTorch/README.md) · [Notebook](Diabetes%20Risk%20Prediction%20with%20PyTorch/starter-kit/diabetes_prediction_mlp.ipynb) |
| Landmark classification | Predict landmarks in photographs | CNN from scratch, transfer learning, top-k inference | [Project README](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/README.md) · [Notebooks](#2-landmark-classification-and-tagging) |
| Transformer-based Q&A | Compare document retrievers and demonstrate RAG | BM25, Word2Vec, SentenceTransformers, information retrieval metrics | [Project README](Build%20a%20Transformer-Based%20Q%26A%20System%20Project/README.md) · [Notebook](Build%20a%20Transformer-Based%20Q%26A%20System%20Project/notebooks/unified_retrieval_comparison.ipynb) |
| Synthetic handwriting | Generate and evaluate class-conditioned MNIST digits | cGAN, conditional U-Net, DDPM/DDIM, FID | [Project README](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/README.md) · [Notebooks](#4-synthetic-handwriting-generator) |

## 1. Diabetes risk prediction

The [diabetes project](Diabetes%20Risk%20Prediction%20with%20PyTorch/) builds
an MLP using CDC health indicators. The notebook explores the data, creates
stratified train/validation/test splits, scales features, trains and tunes
models, and evaluates classification metrics and decision thresholds.

The saved [test results](Diabetes%20Risk%20Prediction%20with%20PyTorch/starter-kit/diabetes_test_results.csv)
report **89.17% recall**, **67.90% precision**, and **0.8207 ROC AUC** for the
selected model at a threshold of approximately 0.388. These are results on
the project's test split, not evidence of clinical deployment readiness.
Performance and predictive values can differ with population prevalence and
other real-world conditions. This project is educational and is **not medical
advice or a diagnostic device**.

To explore it, open
[diabetes_prediction_mlp.ipynb](Diabetes%20Risk%20Prediction%20with%20PyTorch/starter-kit/diabetes_prediction_mlp.ipynb)
from its `starter-kit/` directory and run the cells in order. The dataset is
in its `data/` directory. See the project README for dependencies; its pinned
PyTorch packages target CUDA 12.1 and may need platform-specific alternatives.

## 2. Landmark classification and tagging

The [landmark project](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/)
classifies images among 50 landmark categories and demonstrates top-k
predictions for photographs. It compares a CNN trained from scratch with
transfer learning.

![Example image used in the landmark classification project](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/static_images/sample_landmark_output.png)

Run the notebooks in order:

1. [CNN from scratch](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/Project_Landmarks_Part1_CNNfromScratch__starter.ipynb)
2. [Transfer learning](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/Project_Landmarks_Part2_TransferLearning__starter.ipynb)
3. [Prediction app](Landmark%20Classification%20%26%20Tagging%20for%20Social%20Media%20Project/Project_Landmarks_Part3_App__starter.ipynb)

The saved notebook outputs report **744/1,250 (59.52%)** test accuracy for the
scratch CNN and **1,031/1,250 (82.48%)** for transfer learning. These reflect
the saved runs; rerunning the current code may produce different results.
The images are a subset of Google Landmarks Dataset v2. Training requires the
project dataset and benefits from a GPU. The project README contains historical
starter setup instructions; use the notebook filenames linked above. Its
`requirements.txt` uses nonstandard dependency notation rather than pip
requirements syntax, so adapt the listed packages to your environment.

## 3. Transformer-based Q&A system

The [retrieval project](Build%20a%20Transformer-Based%20Q%26A%20System%20Project/)
compares keyword-based BM25, corpus-trained Word2Vec, and pretrained
SentenceTransformers. The comparison notebook reports Recall@k, Precision@k,
MRR, timings, example rankings, and a PCA visualization. A Streamlit demo
connects retrieval to optional answer generation.

In the saved small fictional-corpus run, held-out MRR was **0.8021** for BM25,
**0.8750** for Word2Vec, and **0.7646** for MiniLM. The corpus contains only
five documents, so these results demonstrate the workflow rather than establish
a general ranking of retrieval methods. Larger optional models and the BEIR
Natural Questions dataset can be explored in the notebook.

From the project directory, install its dependencies and run the
[comparison notebook](Build%20a%20Transformer-Based%20Q%26A%20System%20Project/notebooks/unified_retrieval_comparison.ipynb).
To launch the demo:

```bash
cd "Build a Transformer-Based Q&A System Project"
python -m pip install -r requirements.txt
cd streamlit
streamlit run app.py
```

Retrieval does not require an API key. Answer generation is optional and,
when enabled, sends the query and selected passages to an external provider.
Configure credentials locally as described in the project README and never
commit them. Model weights may download on first use.

## 4. Synthetic handwriting generator

The [handwriting project](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/)
trains a class-conditional GAN and conditional U-Net diffusion models on
MNIST. It compares generated images using Inception FID and trains classifiers
on synthetic digits before testing them on real MNIST images. It also includes
latent-space walks, linear-versus-cosine schedules, a DDIM sampling-step sweep,
and a four-digit CAPTCHA-style demonstration.

The saved evaluation reports:

| Model | FID (lower is better) | Synthetic-only classifier accuracy on real MNIST test |
|---|---:|---:|
| cGAN | 8.1374 | 95.52% |
| Linear diffusion | 8.1670 | 97.20% |
| Cosine diffusion | 8.2718 | Not reported separately |

FID uses 5,000 generated samples per model and an MNIST **training-set**
reference, not a held-out generative benchmark. Results depend on the trained
checkpoints and evaluation settings. The CAPTCHA is a visual demonstration,
not a secure human-verification system.

From the project directory, install its dependencies and run the four notebooks
in order:

1. [Data preparation](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/starter/00_data_preparation.ipynb)
2. [cGAN training](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/starter/01_cGAN_training.ipynb)
3. [Diffusion training](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/starter/02_diffusion_training.ipynb)
4. [Evaluation](Synthetic%20Handwriting%20Generator%20for%20CAPTCHA%20Systems%20Project/starter/03_evaluation.ipynb)

The evaluation notebook needs the checkpoints produced by both training
notebooks. MNIST and Inception weights may download on first use; a CUDA GPU
is recommended for training and evaluation. See the project README for setup,
checkpoint paths, and the CAPTCHA command.

## Working with the repository

- Each project has its own dependencies; use the corresponding project
  directory and README rather than assuming one environment fits all four.
- Open notebooks with a Jupyter kernel using the environment where their
  dependencies are installed. Run cells in order.
- Training and model downloads can require substantial time, disk space, and
  network access. Saved results are examples, not guarantees for a new run.
- Tests, where provided, are optional for running notebooks but useful for
  validating changes. Keep datasets, checkpoints, and credentials out of
  commits when excluded by the relevant project configuration.
