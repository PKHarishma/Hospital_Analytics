from sentence_transformers import SentenceTransformer
from Chatbot.load_sql import engine
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from sqlalchemy import text
from pathlib import Path
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


db = SQLDatabase(engine)

print("Database connected successfully!")


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("AI connected successfully!")


document_path = (
    Path(__file__).resolve().parent
    / "document"
    / "HospitalAnalytics.pdf"
)

reader = PdfReader(document_path)

pdf_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        pdf_text += page_text + "\n"

print("Total pages:", len(reader.pages))
print("Extracted characters:", len(pdf_text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(pdf_text)

print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

print("Embeddings created successfully!")
print("Embedding shape:", embeddings.shape)


chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="hospital_information"
)


if collection.count() == 0:
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

print("Documents stored in ChromaDB!")


def ask_hospital(question):

    question_lower = question.lower()

    print("\nQuestion received:", question_lower)


    sql_keywords = [
        "how many",
        "how much",
        "average",
        "avg",
        "total",
        "count",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "best",
        "worst",
        "top",
        "bottom",
        "most",
        "least",
        "ranking",
        "revenue",
        "billing",
        "patients",
        "patient",
        "admissions",
        "admission",
        "gender",
        "department",
        "doctor",
        "doctors",
        "insurance",
        "age",
        "stay",
        "length of stay"
    ]


    matched_keywords = [
        keyword
        for keyword in sql_keywords
        if keyword in question_lower
    ]

    print("SQL keywords found:", matched_keywords)


    if matched_keywords:

        route = "SQL"

    else:

        router_prompt = f"""
You are a question router for a Hospital Analytics system.

Decide whether the user's question should be answered using RAG or SQL.

Use SQL when the question requires:

- Counting
- Calculating
- Ranking
- Finding highest or lowest
- Finding best or worst
- Finding top or bottom
- Comparing values
- Filtering patient records
- Aggregation
- Patient statistics
- Revenue calculations
- Admission statistics

Use RAG when the question asks about:

- Definitions
- Explanations
- Meaning of hospital terms
- General hospital information
- What something means
- Information that does not require calculation

Examples:

"How many patients are there?" → SQL
"How many patients have diabetes?" → SQL
"What is the average billing?" → SQL
"Which department has the highest revenue?" → SQL
"Which doctor has the most patients?" → SQL
"Who is the best doctor?" → SQL
"Who is the top doctor?" → SQL
"Which doctor treated the most patients?" → SQL

"What is elective admission?" → RAG
"What is emergency admission?" → RAG
"What does length of stay mean?" → RAG
"What is cardiology?" → RAG

Return ONLY one word:

RAG

or

SQL

User question:
{question}
"""

        route_response = llm.invoke(
            router_prompt
        )

        route = route_response.content.strip().upper()


    print("Route:", route)


    if route == "RAG":

        question_embedding = embedding_model.encode(
            [question]
        )

        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=4
        )

        relevant_chunks = results["documents"][0]

        print("\nRetrieved chunks:")

        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i + 1}:")
            print(chunk)

        context = "\n".join(relevant_chunks)


        rag_prompt = f"""
You are a helpful hospital analytics assistant.

The information below comes from the Hospital Analytics
Power BI dashboard.

Use ONLY the information provided below to answer the
user's question.

The user may ask the same information using different words.
Understand the meaning of the question.

Hospital dashboard information:

{context}

User question:

{question}

If the dashboard information contains the answer,
give the answer clearly and directly.

If the provided information does not contain the answer,
say that you don't have enough information.

Do not invent information.

Use simple English.
"""

        response = llm.invoke(
            rag_prompt
        )

        return response.content


    elif route == "SQL":

        sql_prompt = f"""
You are an expert Microsoft SQL Server analyst.

Generate SQL for the table:

Patients

Columns:

PatientID,
Name,
Age,
Gender,
BloodType,
MedicalCondition,
DateOfAdmission,
Doctor,
Hospital,
InsuranceProvider,
BillingAmount,
RoomNumber,
AdmissionType,
DischargeDate,
Medication,
TestResults,
Department,
AgeGroup,
LengthOfStay,
AdmissionYear,
AdmissionMonth,
AdmissionMonthName

User question:

{question}

Rules:

- Return ONLY SQL.
- Do not use markdown.
- Do not explain.
- Use Microsoft SQL Server syntax.
- Do not use LIMIT.
- Use TOP when necessary.
- Use the exact column names provided above.
- Do not invent columns.

When counting records:

Use COUNT(*).

When calculating an average:

Use AVG().

When calculating total revenue or billing:

Use SUM(BillingAmount).

When finding the highest value:

Use ORDER BY ... DESC.

When finding the lowest value:

Use ORDER BY ... ASC.

When the user asks for the best doctor,
top doctor, doctor with the most patients,
or doctor who treated the most patients:

Group by Doctor,
count the patients,
order by patient count descending,
and return the top doctor.

When the user asks for revenue by department:

Group by Department and calculate
SUM(BillingAmount).

Return all departments unless the user explicitly
asks for only the highest or lowest department.

When the user asks which department has the
highest revenue:

Group by Department,
calculate SUM(BillingAmount),
order by revenue descending,
and return the top department.

When the user asks for patient counts by category:

Use COUNT(*) and GROUP BY.

When the user asks about gender:

Use Gender and GROUP BY Gender when appropriate.

When the user asks about admission type:

Use AdmissionType and GROUP BY AdmissionType when appropriate.

When the user asks about medical conditions:

Use MedicalCondition and GROUP BY MedicalCondition when appropriate.

Return only executable SQL.
"""

        sql_response = llm.invoke(
            sql_prompt
        )

        sql_query = sql_response.content.strip()

        sql_query = sql_query.replace(
            "```sql",
            ""
        )

        sql_query = sql_query.replace(
            "```",
            ""
        )

        sql_query = sql_query.strip()


        print("\nGenerated SQL:")
        print(sql_query)


        try:

            with engine.connect() as connection:

                result = connection.execute(
                    text(sql_query)
                )

                rows = result.fetchall()

                column_names = list(
                    result.keys()
                )


            print("\nSQL Result:")
            print(rows)

            print("\nColumns:")
            print(column_names)


            answer_prompt = f"""
You are a helpful hospital data analyst.

User question:

{question}

SQL query used to answer the question:

{sql_query}

Columns returned by the query:

{column_names}

SQL result:

{rows}

Give the final answer to the user's question based
ONLY on the SQL query and its result.

Important:

Understand what each returned column represents
by looking at the SQL query and column names.

Do not guess the meaning of a value.

COUNT means a number of records or patients.

SUM means a total.

AVG means an average.

MAX means the maximum value.

MIN means the minimum value.

GROUP BY means the results are separated by category.

TOP 1 means the result represents the top result
according to the ORDER BY condition.

Use the column names to understand the meaning
of every value.

For example, if the query returns:

Doctor, PatientCount

and the result is:

Michael Smith, 27

then 27 means the number of patients,
not the doctor's age.

If the query returns:

Department, TotalRevenue

then the value represents revenue.

If the query returns:

MedicalCondition, PatientCount

then the value represents the number of patients
with that medical condition.

If the query returns multiple rows,
summarize them clearly.

If the result is empty,
say that no matching data was found.

Do not invent information.

Rules:

- Use simple English.
- Answer directly.
- Do not mention SQL.
- Do not mention Python.
- Do not mention the database.
"""

            answer_response = llm.invoke(
                answer_prompt
            )

            return answer_response.content


        except Exception as e:

            print("SQL Error:", e)

            return (
                f"Sorry, I couldn't process that question. "
                f"Error: {e}"
            )


    else:

        return "I could not determine how to answer that question."