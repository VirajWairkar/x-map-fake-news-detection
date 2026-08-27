# X-MAP: Explainable Multimodal Fake News Detection

## Overview

**X-MAP** is a multimodal machine learning system for detecting whether a social media post is **Real or Fake** by combining information from both **text and images**.

The model uses:

* **DistilBERT** for extracting text features.
* **ResNet50** for extracting image features.
* A **fusion module** to combine text and image representations.
* A classification head to predict **Real/Fake**.
* **LIME** for explaining important text words.
* **Grad-CAM** for visualizing important image regions.

The improved implementation focuses on solving class imbalance, incorrect image loading, overfitting, and unreliable explainability.

## Key Improvements

The original implementation achieved about **59.7% accuracy** but failed to identify the Fake class effectively. The improved version introduces several changes:

1. **Correct image loading**

   * Images are loaded using the Fakeddit submission `id`.
   * This fixes the previous issue where images were incorrectly loaded using their row index.

2. **Class-weighted loss**

   * Helps the model learn both Real and Fake classes instead of favoring the majority class.

3. **Train/Validation/Test split**

   * The dataset is divided into training, validation, and test sets using stratification.

4. **Best model checkpoint**

   * The model with the best validation F1-score is saved and used for final testing.

5. **Early stopping**

   * Helps reduce overfitting during training.

6. **Differential learning rates**

   * Pretrained encoders and the newly initialized classifier use different learning rates.

7. **Partial layer freezing**

   * Earlier layers of DistilBERT and ResNet50 are frozen to preserve pretrained representations and reduce overfitting.

8. **Image augmentation**

   * Training images use random cropping, horizontal flipping, and color jitter.
   * Validation and test images use deterministic preprocessing.

9. **Improved LIME explanation**

   * LIME explains text while keeping the actual image associated with the sample fixed.
   * This provides an explanation of the real multimodal prediction rather than a prediction using a blank image.

## Model Architecture

```text
                 Input Social Media Post
                         |
              +----------+----------+
              |                     |
            Text                  Image
              |                     |
         DistilBERT              ResNet50
              |                     |
        Text Features          Image Features
              |                     |
              +----------+----------+
                         |
                   Feature Fusion
                         |
                  Classification
                         |
                +--------+--------+
                |                 |
              Real              Fake
```

## Dataset

The implementation uses the **Fakeddit** dataset through `test_public.csv`.

The notebook:

* Loads the dataset from `test_public.csv`.
* Uses the `clean_title` column as the text input.
* Uses the `id` column to locate the corresponding image.
* Uses `2_way_label` as the binary classification label.
* Currently works with a configurable subset of **15,000 samples**.

Images are downloaded from their `image_url` when required and stored in the `fakeddit_images/` directory.

## Data Split

The selected dataset is split into:

* **70% Training**
* **10% Validation**
* **20% Testing**

Stratified splitting is used to preserve the Real/Fake class distribution.

## Technologies Used

* Python
* PyTorch
* Torchvision
* Hugging Face Transformers
* DistilBERT
* ResNet50
* Scikit-learn
* LIME
* Grad-CAM
* Pandas
* PIL
* Matplotlib

## Installation

Install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install transformers pandas scikit-learn pillow requests
pip install lime grad-cam matplotlib
```

A CUDA-enabled GPU is recommended for training. The notebook is configured for a **T4 GPU** in Google Colab.

## Project Structure

```text
X-MAP/
│
├── X_MAP_improved.ipynb
├── test_public.csv
├── fakeddit_images/
│   ├── image_id_1.jpg
│   ├── image_id_2.jpg
│   └── ...
│
├── xmap_model.pth
└── xmap_model_best.pth
```

## How to Run

### 1. Open the Notebook

Open:

```text
X_MAP_improved.ipynb
```

in Google Colab or Jupyter Notebook.

### 2. Prepare the Dataset

Place `test_public.csv` in the working directory.

If the required images are not already available, run the image-download cell. Images are saved using their Fakeddit IDs.

### 3. Create the Dataset

The notebook creates separate datasets for:

```text
Training
Validation
Testing
```

Training uses augmented images, while validation and testing use clean deterministic transformations.

### 4. Train the Model

Run the training cells to fine-tune the X-MAP architecture.

The best-performing model is saved as:

```text
xmap_model_best.pth
```

and subsequently copied to:

```text
xmap_model.pth
```

### 5. Evaluate the Model

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

The final evaluation uses the **best validation checkpoint** rather than simply using the final training epoch.

## Explainability

X-MAP provides two forms of explainability.

### LIME – Text Explanation

LIME identifies words in the post that contribute to the prediction.

Example:

```text
Input Text
    ↓
LIME
    ↓
Important Words
    ↓
Real/Fake Contribution
```

The improved implementation keeps the actual image fixed while LIME perturbs the text, making the explanation consistent with the multimodal prediction.

### Grad-CAM – Image Explanation

Grad-CAM identifies important regions of the image that influence the model's prediction.

```text
Input Image
    ↓
ResNet50
    ↓
Grad-CAM
    ↓
Important Image Regions
```

The target layer used for visual explanation is the final convolutional block of ResNet50.

## Modular Pipeline

The notebook also separates the system into reusable components:

```text
TextEncoder
     ↓
ImageEncoder
     ↓
FusionModule
     ↓
ClassificationHead
     ↓
ExplanationModule
```

This makes the X-MAP pipeline easier to understand and reuse.

## Output

For a given sample, the system produces:

```text
Prediction: Real / Fake
Confidence: <probability>
```

It can also generate:

* Text-based LIME explanations.
* Image-based Grad-CAM visualizations.

## Model Files

After training, the model weights are stored as:

```text
xmap_model_best.pth
xmap_model.pth
```

The best checkpoint is selected according to validation F1-score.

## Important Notes

* The image files must be named according to their Fakeddit `id`.
* The notebook expects the image directory to be named `fakeddit_images`.
* GPU acceleration is recommended for practical training times.
* The number of training samples can be changed using `NUM_SAMPLES`.
* The current notebook configuration uses 15,000 samples.

## Future Improvements

Possible future improvements include:

* Training on the complete dataset.
* Hyperparameter optimization.
* More advanced multimodal fusion techniques.
* Improved handling of posts without images.
* Additional evaluation metrics.
* Deployment as a web or mobile application.
* Real-time Fake/Real prediction.
* More comprehensive multimodal explainability.

## Conclusion

X-MAP demonstrates an **explainable multimodal approach to Fake/Real classification** by combining textual and visual information. The improved implementation addresses important issues in the original pipeline, including incorrect image loading, class imbalance, overfitting, validation, and multimodal explanation reliability.

The resulting system provides not only a prediction but also explanations showing **which text and image features contributed to the decision**.
