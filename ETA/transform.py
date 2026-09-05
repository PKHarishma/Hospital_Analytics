

import pandas as pd
from pathlib import Path

data_dir= Path(__file__).resolve().parent.parent/"data"

data_dir.mkdir(parents=True, exist_ok=True)

RAW_File = data_dir/ "raw_healthcare_dataset.csv"
CLEANED_File =data_dir/ "clean_healthcare_dataset.csv"


def load_raw_data(path:Path=RAW_File)->pd.DataFrame:
    df=pd.read_csv(path)
    print("Columns in RAW file:")
    print(df.columns.tolist())

    print("Department exists:", "Department" in df.columns)
    print(df.shape) # checking no of rows and columns
    print(df.isnull().sum()) # checking missing values of all columns
    print(df.describe(include='all'))
    return df

def remove_duplicates(df:pd.DataFrame)->pd.DataFrame:
    before=len(df)
    df=df.drop_duplicates()
    print(f"{before-len(df) } duplicates removed")
    return df

def handle_missing_values(df:pd.DataFrame)->pd.DataFrame:
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Billing Amount"] = df["Billing Amount"].fillna(df["Billing Amount"].median())
    df["Insurance Provider"] = df["Insurance Provider"].fillna("Unknown")

    # Drop rows missing critical identifying fields
    critical_cols = ["Name", "Date of Admission", "Doctor", "Hospital"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"Dropped {before - len(df)} rows missing critical fields")
    return df

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df["Age"] = df["Age"].astype(int)
    df["Billing Amount"] = df["Billing Amount"].astype(float).round(2)
    df["Room Number"] = df["Room Number"].astype(int)
    return df


def validate_billing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["Billing Amount"] > 0]
    print(f"Removed {before - len(df)} rows with invalid (non-positive) billing")
    return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 12, 19, 35, 50, 65, 120]
    labels = ["Child (0-12)", "Teen (13-19)", "Young Adult (20-35)",
              "Adult (36-50)", "Middle Age (51-65)", "Senior (65+)"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df


def add_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df["Length of Stay"] = df["Length of Stay"].clip(lower=0)
    return df


def add_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["Admission Year"] = df["Date of Admission"].dt.year
    df["Admission Month"] = df["Date of Admission"].dt.month
    df["Admission Month Name"] = df["Date of Admission"].dt.strftime("%B")
    return df


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "PatientID", range(1, len(df) + 1))
    return df
def changing_colums(df:pd.DataFrame)->pd.DataFrame:
    df = df.rename(columns={
        "Billing Amount": "BillingAmount",
        "Admission Year": "AdmissionYear",
        "Admission Month": "AdmissionMonth",
        "Admission Month Name": "AdmissionMonthName",
        "Age Group": "AgeGroup",
        "Length of Stay": "LengthOfStay",
        "Insurance Provider": "InsuranceProvider",
        "Medical Condition": "MedicalCondition",
        "Blood Type": "BloodType",
        "Date of Admission": "DateOfAdmission",
        "Discharge Date": "DischargeDate",
        "Room Number": "RoomNumber",
        "Admission Type": "AdmissionType",
        "Test Results": "TestResults"
    })
    return df

def run_transform() -> pd.DataFrame:
    df = load_raw_data()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = validate_billing(df)
    df = add_age_groups(df)
    df = add_length_of_stay(df)
    df = add_calendar_fields(df)
    df = add_patient_id(df)
    df=changing_colums(df)
    df.to_csv(CLEANED_File, index=False)
    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Cleaned dataset written to: {CLEANED_File}")
    print(df.columns)
    return df


run_transform()