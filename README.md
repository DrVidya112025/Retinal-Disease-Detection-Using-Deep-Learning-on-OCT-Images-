# Retinal Disease Detection Using Deep Learning on OCT Images

Classifying retinal Optical Coherence Tomography (OCT) scans into four categories with NumPy and PyTorch, from raw data to an explained model.

| Class | Meaning |
|---|---|
| **CNV** | Choroidal neovascularization: abnormal vessel growth under the retina, with fluid or membrane |
| **DME** | Diabetic macular edema: retinal thickening and fluid caused by diabetes |
| **DRUSEN** | Yellow deposits under the retina, an early sign of age-related macular degeneration |
| **NORMAL** | Healthy retina with a preserved foveal contour |

## Results

All models were evaluated on the same 1,000 held-out test scans (250 per class, 112 x 112 px), which were never used for training, tuning or model selection.

| Model | Params (M) | Accuracy | Macro F1 |
|---|---|---|---|
| **ResNet18 (tuned)** | 11.19 | **0.940** | **0.938** |
| MobileNetV2 | 2.23 | 0.935 | 0.934 |
| VGG16 | 14.85 | 0.830 | 0.830 |
| DenseNet121 | 6.96 | 0.793 | 0.790 |
| Custom CNN | 0.52 | 0.753 | 0.750 |

Highlights:

- Pretrained CNNs fine-tuned on a small balanced subset beat the from-scratch CNN by about 19 points.
- MobileNetV2 is almost as accurate as ResNet18 at one fifth of the size.
- DRUSEN is the hardest class in every model (ResNet18 recall 0.784, often confused with CNV).
- Grad-CAM heatmaps are coarse and only partly faithful in the deletion test (see Step 8).

## Project structure

The work is split into eight notebooks, meant to be run in order.

| Step | Notebook | What it covers |
|---|---|---|
| 1 | `Step1_Dataset_Collection_and_Understanding.ipynb` | Load data, labels, class balance, image properties |
| 2 | `Step2_Image_Preprocessing.ipynb` | Resize, tensors, normalization, augmentation, stratified splits |
| 3 | `Step3_Neural_Network_Fundamentals.ipynb` | Perceptron and MLP in NumPy, activations, optimizers, regularization |
| 4 | `Step4_Custom_CNN_PyTorch2.ipynb` | Custom CNN, activation/pooling/loss/batch-size/learning-rate experiments |
| 5 | `Step5_Transfer_Learning_1.ipynb` | ResNet18, VGG16, MobileNetV2, DenseNet121: feature extraction vs fine-tuning |
| 6 | `Step6_Model_Optimization_Regularization.ipynb` | Over/underfitting, dropout, L2, BatchNorm, early stopping, random search, 3 seeds |
| 7 | `Step_7_Model_Evaluation.ipynb` | Five models on one test set, confusion matrices, per-class metrics |
| 8 | `Step8_Model_Explainability_GradCAM.ipynb` | Grad-CAM, tissue-focus check, deletion and randomization tests |

## Dataset

- **Source:** Kermany et al. (2018) OCT2017, loaded from Hugging Face as `zacharielegault/Kermany2017-OCT`.
- **Size:** 108,309 training scans and a balanced 1,000-scan test split (250 per class). The test split is patient-independent.
- **Training class counts:** NORMAL 51,140, CNV 37,205, DME 11,348, DRUSEN 8,616.
- **Format:** grayscale JPEG, heights about 496 px, widths 384 to 1,536 px.
- Step 1 explores a small 480-image mirror (`Augusto777/OCT2017`) for a quick structure check; later steps use the full dataset above.

## Method summary

- **Preprocessing:** resize to a square (224 px, later 64 and 112 px for speed), grey channel repeated to 3 for pretrained models, ImageNet normalization, train-only augmentation (horizontal flip, rotation up to 10 degrees, 5% shift, 0.9 to 1.1 zoom, brightness and contrast 20%). No vertical flip.
- **Class imbalance:** the first custom CNN (Step 4) never predicted DRUSEN. Steps 5 to 8 use balanced training pools (510 to 600 scans per class).
- **Transfer learning:** the 1,000-way output layer is replaced by a 4-way layer. Feature extraction and partial fine-tuning are compared.
- **Tuning:** random search plus 3 seeds per configuration to tell real gains from noise.
- **Explainability:** Grad-CAM with two sanity checks (deletion test and last-layer randomization).

## Getting started

The notebooks were run in Google Colab (GPU for Steps 5 to 7) and also work locally.

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install torch torchvision datasets pillow matplotlib seaborn pandas numpy
jupyter notebook
```

Run the notebooks in order. Later steps read files written by earlier ones:

- `preprocessing_config.json` (Step 2)
- `step6_best_model.pt` and `optimization_config.json` (Step 6), needed by Steps 7 and 8
- Step 7 also expects checkpoints for the custom CNN, MobileNetV2, VGG16 and DenseNet121

Some cells use `QUICK_MODE = True` (smaller images, shorter runs). Set it to `False` for larger, slower runs, ideally on a GPU.

## Known limitations

- **Single dataset and device.** Results may not transfer to other scanners or hospitals.
- **Small, short experiments.** Most pretrained runs use about 2,000 balanced scans at 112 px with few seeds, so differences of one or two points are within noise.
- **Possible patient overlap.** The Hugging Face copy has no patient IDs, so validation scores may be optimistic. Trust the test split.
- **Grad-CAM model.** If `step6_best_model.pt` and `optimization_config.json` are not next to the Step 8 notebook, it trains a quick stand-in ResNet18 (about 84.8% test accuracy) and explains that instead. The saved outputs in the notebook come from this stand-in.
- **Inconsistent numbers between notebooks.** Step 5 reports 96.2% test accuracy for a single VGG16 full fine-tune, while the Step 7 retrained VGG16 reaches 83.0%. Step 7 first shows 0.888 for ResNet18 on a fallback test subset, then 0.940 on the cached test set. Use the Step 7 comparison table for final numbers.
- **Not for clinical use.** This is a research and learning project, not a validated diagnostic tool.

## References

- Kermany, D. S. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122-1131.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

## Author

Vidya Priya V
