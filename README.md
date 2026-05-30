# Pneumonia Detection Website (DenseNet121)

This project serves your trained model `best_densenet122.pth` through a working Flask website and API for chest X-ray classification.

Important note: the filename says `densenet122`, but the architecture used is `torchvision.models.densenet121` (as trained in your notebook).

## What is included

- `app.py`: Flask web server
- `inference_pipeline.py`: reusable PyTorch inference pipeline
- `templates/index.html`: upload + prediction page
- `static/style.css`: UI styles
- `requirements.txt`: dependencies
- `model_building.ipynb`: your original training notebook
- `best_densenet122.pth`: trained model weights used by the app

## Model summary

- **Architecture**: DenseNet121 (`torchvision.models.densenet121`)
- **Pretraining**: ImageNet initialization in training notebook (`weights="IMAGENET1K_V1"`)
- **Classifier head**: replaced with `Linear(1024 -> 2)` for:
  - `NORMAL`
  - `PNEUMONIA`
- **Saved checkpoint**: state dict (`model.state_dict()`)

## Dataset summary

From your local `data/` folders:

- Train: `5216` images
- Validation: `16` images
- Test: `624` images

Class mapping:

- `NORMAL` -> `0`
- `PNEUMONIA` -> `1`

Class distribution:

- Train: `NORMAL=1341`, `PNEUMONIA=3875`
- Validation: `NORMAL=8`, `PNEUMONIA=8`
- Test: `NORMAL=234`, `PNEUMONIA=390`

## Training pipeline (from `model_building.ipynb`)

### 1) Data transforms

- **Train**
  - `Resize((224, 224))`
  - `RandomRotation(10)`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Validation / Test**
  - `Resize((224, 224))`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### 2) DataLoader setup

- `batch_size=2`
- `num_workers=2`
- `shuffle=True` for train, `False` for val/test

### 3) Optimization setup

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `1e-5`
- Epochs: `100`
- Best model policy: save checkpoint when validation accuracy improves

### 4) Reported results in notebook output

- Best validation accuracy: `1.0000`
- Test accuracy: `0.8974`
- Confusion matrix:
  - `[[174, 60], [4, 386]]`
- Classification report:
  - NORMAL: Precision `0.98`, Recall `0.74`, F1 `0.84`
  - PNEUMONIA: Precision `0.87`, Recall `0.99`, F1 `0.92`
  - Weighted F1: `0.89`

## Inference pipeline used by the website

`inference_pipeline.py` does:

1. Loads `best_densenet122.pth`
2. Builds DenseNet121 with `Linear(1024 -> 2)` classifier
3. Applies the same validation/test preprocessing pipeline
4. Runs softmax and returns:
   - predicted class
   - confidence
   - per-class probabilities

## Run the website

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

- `http://127.0.0.1:5000`

## API usage

Endpoint:

- `POST /api/predict`
- form-data key: `image`

Example:

```bash
curl -X POST \
  -F "image=@data/test/NORMAL/IM-0001-0001.jpeg" \
  http://127.0.0.1:5000/api/predict
```

Example JSON response:

```json
{
  "ok": true,
  "predicted_index": 0,
  "predicted_class": "NORMAL",
  "confidence": 0.92,
  "probabilities": {
    "NORMAL": 0.92,
    "PNEUMONIA": 0.08
  }
}
```

## Configuration

Optional environment variables:

- `MODEL_PATH` (auto-detects `best_dencenet122.pth` or `best_densenet122.pth`; fallback is `best_densenet122.pth`)
- `TRAIN_DIR` (default: `data/train`)
- `MAX_UPLOAD_MB` (default: `10`)

Example:

```bash
MODEL_PATH=best_densenet122.pth MAX_UPLOAD_MB=15 python app.py
```

## Limitations and practical notes

- Validation set is very small (`16` images), so validation accuracy is not very stable.
- Dataset is imbalanced toward `PNEUMONIA`.
- This is a research/demo classifier and **not** a medical diagnostic device.

## Next improvements you can do

- Increase validation set size (or use k-fold validation).
- Add `RandomHorizontalFlip`, contrast/brightness augmentations.
- Track AUC/ROC and calibration in addition to accuracy.
- Export model to TorchScript or ONNX for lighter deployment.
# -Pneumonia-Detection-
