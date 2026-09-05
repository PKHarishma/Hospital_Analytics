import pandas as pd
import pickle


# 1. Load trained model
with open("../models/provider_model.pkl", "rb") as file:
    artifact = pickle.load(file)

model = artifact["model"]
feature_columns = artifact["feature_columns"]


# 2. Load cleaned data
df = pd.read_csv("../data/clean_healthcare_dataset.csv")


# 3. Select the same features used during training
X = df[
    ["Age", "Gender", "MedicalCondition", "AdmissionType"]
]


# 4. Convert categorical columns into numbers
X = pd.get_dummies(X)


# 5. Make columns match the training data
X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


# 6. Predict 0 or 1
predictions = model.predict(X)


# 7. Get probability of Long Stay
probabilities = model.predict_proba(X)[:, 1]


# 8. Add predictions to dataframe
df["Predicted_Long_Stay"] = predictions
df["Risk_Probability"] = probabilities


# 9. Convert 0/1 into meaningful labels
df["Provider_Performance_Label"] = df[
    "Predicted_Long_Stay"
].map({
    0: "Stable",
    1: "High Burden"
})


# 10. Display results
print(
    df[
        [
            "Age",
            "Gender",
            "MedicalCondition",
            "AdmissionType",
            "Predicted_Long_Stay",
            "Risk_Probability",
            "Provider_Performance_Label"
        ]
    ].head(10)
)