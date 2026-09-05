from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

host = "aws-0-ap-northeast-2.pooler.supabase.com"
port = 5432
database = "postgres"
username = "postgres.teglhtzfgvfjdbxjukwu"
password = os.getenv("SUPABASE_PASSWORD")

if not password:
    raise ValueError("SUPABASE_PASSWORD is not configured")

engine = create_engine(
    f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
)

with engine.connect() as connection:
    result = connection.execute(
        text('SELECT COUNT(*) FROM "Patients"')
    )
    count = result.scalar()

print("Supabase database connected successfully!")
print("Total patients:", count)