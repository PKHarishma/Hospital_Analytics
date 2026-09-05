import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# 1. Load CLEANED data
df = pd.read_csv("../data/clean_healthcare_dataset.csv")


# 2. Create target
# 1 = Long Stay
# 0 = Normal Stay

median_stay = df["LengthOfStay"].median()

df["Is_Long_Stay"] = (
    df["LengthOfStay"] > median_stay
).astype(int)


# 3. Select features

X = df[
    [
        "Age",
        "Gender",
        "MedicalCondition",
        "AdmissionType"
    ]
]

y = df["Is_Long_Stay"]


# 4. Convert categorical columns into numbers

X = pd.get_dummies(X)


# 5. Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create model

model = RandomForestClassifier(
    random_state=42
)


# 7. Train model

model.fit(X_train, y_train)


# 8. Save model

artifact = {
    "model": model,
    "feature_columns": list(X.columns)
}

with open("../models/provider_model.pkl", "wb") as file:
    pickle.dump(artifact, file)


print("Model trained and saved successfully!")