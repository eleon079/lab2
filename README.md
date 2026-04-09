# House Segmentation Service (Flask + Waitress + Docker)

This project replaces the original text sentiment model with a house-segmentation model for aerial imagery.

## 1. Create local environment file

Copy `.env.example` to `.env` and edit values if needed.

## 2. Prepare from Hugging Face

This project downloads an aerial building-segmentation dataset from Hugging Face and converts it into the local training layout expected by the training script.

Generate masks and train/val/test splits:

```bash
python prepare_dataset.py