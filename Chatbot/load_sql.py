import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
driver = os.getenv("DB_DRIVER")

connection_string = (
    f"mssql+pyodbc://@{server}/{database}"
    f"?driver={driver.replace(' ', '+')}"
    f"&trusted_connection=yes"
    f"&TrustServerCertificate=yes"
)

engine = create_engine(connection_string)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "clean_healthcare_dataset.csv"


def load_data():

    df = pd.read_csv(DATA_FILE)

    df.to_sql(
        "Patients",
        engine,
        if_exists="replace",
        index=False
    )

    print("Data successfully loaded into SQL Server!")


if __name__ == "__main__":
    load_data()