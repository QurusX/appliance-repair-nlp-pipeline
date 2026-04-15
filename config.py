import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# File Paths
INPUT_EXCEL = RAW_DATA_DIR / "Q_A_13k.xlsx"
LABELED_DATA = PROCESSED_DATA_DIR / "labeled_repair_data.jsonl"

# Model Paths
CLASSIFIER_SAVE_PATH = MODELS_DIR / "classifier"
GENERATOR_SAVE_PATH = MODELS_DIR / "generator"
NER_SAVE_PATH = MODELS_DIR / "ner"

# Processing Config
LABELING_LIMIT = 1000
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regex Patterns
PART_NUMBER_PATTERN = r'\b(?:PS|W|WP|WR|WH|E)\d+[A-Z0-9]*\b'