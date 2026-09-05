import pandas as pd
from pathlib import Path
import random

RAW_FILE = Path("../data/raw_healthcare_dataset.csv")
OUTPUT_FILE = Path("../data/clean_healthcare_dataset.csv.")


def add_department():
    df = pd.read_csv(RAW_FILE)

    departments = [
        "Cardiology",
        "Neurology",
        "Orthopedics",
        "Pediatrics",
        "Oncology",
        "General Medicine",
        "Emergency"
    ]

    random.seed(42)

    df["Department"] = [
        random.choice(departments)
        for _ in range(len(df))
    ]

    df.to_csv(RAW_FILE, index=False)

    print(df.head())
    print(df.shape)
    print(df.columns)


add_department()
