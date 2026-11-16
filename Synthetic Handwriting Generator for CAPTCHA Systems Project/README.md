# Conditional Handwriting Generation

Train a class-conditional convolutional GAN and conditional U-Net diffusion
models on MNIST, then evaluate generated digits using Inception FID, visual
comparisons and synthetic-only downstream classification.

## Run in your existing environment

From this project directory:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
cd starter
python -m jupyter notebook
```

Run and save these notebooks **in order**:

1. [00_data_preparation.ipynb](starter/00_data_preparation.ipynb): download MNIST,
   normalize pixels to `[-1, 1]`, inspect images and save a CPU-portable sample.
2. [01_cGAN_training.ipynb](starter/01_cGAN_training.ipynb): train the GAN for
   50 epochs, display all ten digit classes and explore a fixed-label latent walk.
3. [02_diffusion_training.ipynb](starter/02_diffusion_training.ipynb): visualize
   forward noising; train separate linear and cosine models for 40 epochs each;
   demonstrate full DDPM reverse sampling and faster DDIM.
4. [03_evaluation.ipynb](starter/03_evaluation.ipynb): compute actual FID scores,
   compare ten digit classes, train synthetic-only classifiers, report real
   test accuracy, generate a four-digit composite, and plot FID versus sampling
   steps.

Use the same interpreter for the notebook kernel as for installed dependencies.
No new virtual environment is required. Model code selects CUDA, then MPS,
then CPU. CUDA uses float16 autocast with gradient scaling; CPU/MPS use float32.
The diffusion training and sampling runs are substantial: a CUDA GPU is
recommended. There are no background services or remote generation APIs.

Diffusion training checks unscaled gradients before clipping. On a recoverable
AMP overflow, GradScaler skips the optimizer update and lowers its scale;
the EMA update is also skipped. Each event is logged, and epoch histories
record `optimizer_steps` and `skipped_batches`. MSE includes all finite forward
passes, including skipped batches. Non-finite loss, non-finite float32 gradients,
eight consecutive overflows, or an epoch with no valid updates still fail
explicitly rather than saving a misleading successful checkpoint. After
updating the training module in a running notebook, restart the kernel and
rerun its setup cells before training again.

MNIST and the standard torch-fidelity Inception weights are downloaded on
first use. Subsequent runs reuse their caches. Zero DataLoader workers is the
reliable notebook default; enable workers explicitly only when beneficial.
Worker mode uses spawn and a bounded prefetch queue.

## Architectures and objectives

- [cgan.py](starter/model/cgan.py): G concatenates a learned class embedding
  with latent noise and upsamples from 7x7 to 28x28. Its final Tanh produces
  `N x 1 x 28 x 28` images in `[-1, 1]`. D concatenates the image with a learned
  spatial class map and returns **raw logits**.
- [train_cgan.py](starter/training/train_cgan.py): a separate discriminator
  step uses real and detached fake BCE, followed by a generator step using
  non-saturating BCE against real targets. `BCEWithLogitsLoss` combines sigmoid
  with BCE stably. This deliberately replaces the starter's sigmoid + BCELoss
  convention; do not apply another sigmoid before the loss.
- [diffusion.py](starter/model/diffusion.py): residual U-Net with GroupNorm,
  configurable sinusoidal time embeddings, class embeddings at every block
  and concatenated encoder/decoder skips. Output is an unrestricted noise
  prediction with exactly the input shape.
- [train_diffusion.py](starter/training/train_diffusion.py): construct
  `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon` and minimize MSE
  between predicted and sampled epsilon. Training uses AdamW, gradient clipping
  and an EMA model for sampling. Compare the scaled linear beta schedule and
  cumulative cosine schedule at the same training horizon and budget.

Epoch-average histories are written to JSON after every epoch. GAN model
checkpoints have distinct filenames; diffusion checkpoints include both raw
and EMA weights, optimizer/scaler states, training schedule and architecture.
Writes use temporary files and atomic replacement. Paths resolve from source
locations, not the current working directory. Final files include:

```text
checkpoints/cgan_generator_epochfinal.pt
checkpoints/cgan_discriminator_epochfinal.pt
checkpoints/diffusion_linear_epochfinal.pt
checkpoints/diffusion_cosine_epochfinal.pt
```

Load with [load_generator](starter/utils/generation.py), which reconstructs the
architecture and applies EMA weights for diffusion. Sampling rejects any
schedule/horizon mismatch. Old starter checkpoints are incompatible with the
new convolutional GAN and do not contain sufficient schedule metadata: retrain
them rather than trying to silently reuse them.

Full DDPM sampling traverses every training index in reverse with posterior
variance and no final noise. DDIM uses a subset of those same indices.
**Reducing sampling steps does not rebuild the training beta schedule.**
By default evaluation uses 50 DDIM model calls with a trained horizon of 1000.
Seeded CPU random generators make sampling reproducible without resetting the
global RNG. GPU kernels can still introduce numerical nondeterminism.

## Evaluation protocol and leakage controls

- Generate 5,000 class-balanced samples per model for full, 2048-dimensional
  **torch-fidelity Inception FID**, using the same sample count/seeds. Convert
  normalized grayscale to RGB `[0, 1]`; Inception performs its standard
  299x299 resize. Stream bounded batches and cache the real feature moments
  once. Feature extraction remains float32; TorchMetrics accumulates float64
  moments. FID runs on CPU or CUDA, not MPS.
- Use a fixed balanced **MNIST train** reference for FID and selection.
  Generators also train on MNIST train, so these are in-distribution metrics,
  not evidence of held-out generative generalization. Inception features are
  ImageNet-based and imperfect for digits. Small-sample FID is biased.
- Compare all ten digit classes with identical fixed grayscale limits.
  Condition labels are not predictions: manually verify recognizable digits.
  Pixel standard deviation measures within-class variation but can be
  inflated by noise; it is not a standalone meaningful-diversity metric.
- Choose the diffusion schedule by reference FID, then generate **10,000 new
  synthetic images per method** (independent seed) for downstream utility.
  Train two fresh identical CNNs from scratch: one on cGAN outputs, one on
  selected diffusion outputs. Neither CNN trains on real images or uses
  pretrained weights. Report accuracy and per-digit accuracy on all **10,000
  real MNIST test images**, once for each final classifier.
- Use real-train reference classification accuracy, never real-test accuracy,
  to select a CAPTCHA source separately for each digit. Selection reflects
  synthetic-data utility, not necessarily maximum visual fidelity.
- Plot FID at DDIM budgets 10, 25, 50 and 100 using the same initial noises,
  reference images and sample counts. Reuse an already measured 50-step point.
  This measures sampling steps K with training horizon T fixed, not a comparison
  of independently retrained models at different T.

Results, histories, checkpoint hashes, figures and the selection manifest are
saved under `outputs/`. The notebook includes data-dependent Markdown analysis
and explicitly distinguishes measured results from interpretation.

### Four-digit composition

After running the evaluation notebook, from `starter/`:

```bash
python generate_captcha.py 0123 --output ../outputs/captcha_0123.png
```

The script loads `outputs/captcha_models.json`, reuses the selected models and
preserves leading zeroes. This is a **CAPTCHA-style visual demo, not a secure
human-verification system**. MNIST digits are easily recognized by OCR.
Production bot defenses require independent security and accessibility work.

## Verification versus submission evidence

Automated tests cover conditioning and gradients, custom U-Net embedding sizes,
skip shapes, exact forward equations, DDPM/DDIM oracle reconstruction, seeded
sampling, metadata mismatch rejection, actual parameter updates, checkpoint
round trips, synthetic-only classifier wiring, FID math/real-stat reuse,
latent endpoints, visualization, notebook syntax and four-digit composition.
Tests do not train models to convergence or establish human-visible fidelity.

The notebooks intentionally contain **no fabricated training outputs, FID
scores or accuracy claims**. To finish the empirical submission requirements,
run full training and evaluation, visually confirm recognizable class-conditional
digits, save all notebook outputs and include the generated checkpoints/results
as required by your submission platform. Never present smoke-test measurements
as trained-model performance. Generated data, checkpoints and outputs are
git-ignored to avoid accidentally committing large artifacts.
