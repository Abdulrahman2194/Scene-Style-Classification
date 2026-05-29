# Interior Scene Style Classification

This project classifies indoor room images into **17 interior design styles** using a CLIP-based feature extraction pipeline and a lightweight MLP classifier. The model is designed for the Kaggle CSE 281 Spring 2026 Scene Style Classification competition.

## Features

* **CLIP ViT-B/16 Backbone**: Uses OpenAI CLIP `ViT-B-16.pt` for image and text feature extraction.
* **Manual CLIP Implementation**: Includes custom CLIP Vision Transformer, text transformer, tokenizer, and checkpoint loading logic.
* **17 Interior Style Categories**: Supports Asian, Boho, Coastal, Contemporary, Craftsman, Eclectic, Farmhouse, French Country, Industrial, Mediterranean, Minimalist, Modern, Scandinavian, Shabby Chic, Southwestern, Tropical, and Victorian.
* **Prompt Ensemble**: Uses multiple text prompts per class to create stronger text-guided style representations.
* **Feature-Based Classification**: Combines image embeddings, text similarity logits, and zero-shot probabilities into one feature vector.
* **MLP Classifier**: Trains a neural network classifier on extracted CLIP features.
* **Stratified K-Fold Training**: Uses 3-fold stratified cross-validation for more stable validation and prediction.
* **Test-Time Augmentation**: Performs inference using multiple image views, including center, horizontal flip, and center crops.
* **Robust Inference**: Generates both normal and TTA-based submission files.

## Supported Interior Styles

* Asian
* Boho
* Coastal
* Contemporary
* Craftsman
* Eclectic
* Farmhouse
* French Country
* Industrial
* Mediterranean
* Minimalist
* Modern
* Scandinavian
* Shabby Chic
* Southwestern
* Tropical
* Victorian

## Installation

Install the required Python libraries:

```bash
pip install torch torchvision numpy pandas pillow scikit-learn regex ftfy
```

## Requirements

```text
torch
torchvision
numpy
pandas
Pillow
scikit-learn
regex
ftfy
```

The code automatically looks for the CLIP checkpoint and BPE vocabulary file. If they are not found locally, it attempts to download:

```text
ViT-B-16.pt
bpe_simple_vocab_16e6.txt.gz
```

If Kaggle internet is disabled, place these files manually in:

```text
/kaggle/working/clip_assets
```

## Dataset Structure

The dataset should follow this structure:

```text
StyleClassificationIndoors/
├── train/
│   ├── asian/
│   ├── boho/
│   ├── coastal/
│   ├── contemporary/
│   ├── craftsman/
│   ├── eclectic/
│   ├── farmhouse/
│   ├── french-country/
│   ├── industrial/
│   ├── mediterranean/
│   ├── minimalist/
│   ├── modern/
│   ├── scandinavian/
│   ├── shabby-chic-style/
│   ├── southwestern/
│   ├── tropical/
│   └── victorian/
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The code also expects:

```text
class_mapping.txt
sample_submission.csv
```

## Usage

Run the main Python file:

```bash
python scene_style_classification.py
```

The script will:

1. Load the dataset paths.
2. Load or download the CLIP checkpoint and tokenizer vocabulary.
3. Build the manual CLIP model.
4. Generate text features using prompt ensembling.
5. Extract CLIP image features from training and test images.
6. Train an MLP classifier using 3-fold stratified cross-validation.
7. Run inference with and without TTA.
8. Save the final submission files.

## Configuration

Important hyperparameters are stored in the `Config` class:

```python
seed = 42
num_classes = 17
image_size = 224
image_batch_size = 64
n_folds = 3
mlp_epochs = 90
mlp_batch_size = 256
mlp_lr = 1e-3
mlp_weight_decay = 1e-3
label_smoothing = 0.05
early_stop_patience = 12
text_logit_scale = 12.0
zero_shot_weight = 0.0
```

TTA views used during inference:

```python
("center", "flip", "crop92", "crop92_flip", "crop86")
```

## Model Architecture

The model pipeline has three main parts:

### 1. CLIP Feature Extractor

The code builds a manual CLIP model using:

* Vision Transformer image encoder
* Text transformer encoder
* Manual BPE tokenizer
* CLIP normalization values
* Official CLIP ViT-B/16 checkpoint weights

### 2. Text Prompt Ensemble

For each class, the model creates multiple prompts such as:

```text
a photo of a {} interior design room
a photo of a {} style room
an indoor room in {} style
a home interior with {} design style
```

The text embeddings are averaged to create one strong class representation per style.

### 3. MLP Classifier

The classifier input combines:

```text
image_features + text_similarity_logits + zero_shot_probabilities
```

The MLP contains:

* LayerNorm
* Linear layers
* BatchNorm
* ReLU activation
* Dropout
* Final classification layer

## Training Features

* AdamW optimizer
* CrossEntropyLoss with label smoothing
* ReduceLROnPlateau scheduler
* Gradient clipping
* Early stopping
* Stratified K-Fold validation
* GPU support with CPU fallback

## Inference

The model performs inference in two ways:

### No TTA

Uses only the center image view and saves:

```text
/kaggle/working/submission_no_tta.csv
```

### TTA

Uses multiple augmented views and saves the main submission:

```text
/kaggle/working/submission.csv
```

## Output

The final output is a Kaggle-compatible submission file:

```text
submission.csv
```

The script also prints prediction counts for each class after saving the submission.

## Results

Validation accuracy is printed for each fold during training. The script also prints the mean validation accuracy after all folds finish.

```text
Fold validation accuracies: 0.48

```

Update this section with your final validation score after running the notebook/script.
