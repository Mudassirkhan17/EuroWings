"""
Train competitor price ensemble on all non-EW rows and export joblib artifacts.
Run once before deploy: python train_and_export.py
"""

from pathlib import Path

from model_utils import export_models, load_competitor_data

DATA_PATH = Path(__file__).parent / "skyscanner_airfare_data.csv"
MODELS_DIR = Path(__file__).parent / "models"


def main():
    print("Loading data...")
    df = load_competitor_data(DATA_PATH)
    print(f"Training rows: {len(df)}")
    export_models(df, MODELS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
