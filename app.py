import psycopg2
import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API key
if not GEMINI_API_KEY:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="NL2SQL",
        user="postgres",
        password="Sahith@123"
    )
    return conn, conn.cursor()

def get_schema_info():
    conn, cursor = get_db_connection()
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """)
    tables = [row[0] for row in cursor.fetchall()]
    schema_info = ""
    for table in tables:
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
        columns = cursor.fetchall()
        schema_info += f"Table: {table}\n"
        for col, dtype in columns:
            schema_info += f"  - {col} ({dtype})\n"
    conn.close()
    return schema_info

SCHEMA = '''
Table: employees
  - id (integer)
  - name (text)
  - age (integer)
  - department (text)
  - salary (integer)
  - join_date (text)
Table: departments
  - id (integer)
  - name (text)
Table: projects
  - id (integer)
  - name (text)
  - department_id (integer)
Table: customers
  - customer_id (integer, primary key, auto-incrementing)
  - first_name (text)
  - last_name (text)
  - email (text, unique)
  - phone_number (text)
  - address (text)
  - city (text)
  - state (text)
  - zip_code (text)
  - registration_date (date)
Table: orders
  - order_id (integer, primary key, auto-incrementing)
  - customer_id (integer, foreign key, references customers(customer_id))
  - order_date (date)
  - total_amount (decimal)
  - status (text)
'''

def get_sql_from_gemini(question):
    import re
    schema = SCHEMA  # Use the static schema
    print("\n[DEBUG] Using Static Schema:\n", schema)
    prompt = f"""
You are an expert in SQL. Given the following database schema, write an SQL query to answer the question. Only use the tables and columns provided in the schema. If the question is not related to the schema, or if the answer cannot be found in the database, respond only with 'no information' and do not write any SQL query. When writing SQL queries, use ILIKE for string comparisons to ensure case-insensitive matching in PostgreSQL. If the question requires data from multiple tables, use appropriate JOINs based on the schema relationships.

Schema:
{schema}

Question:
{question}

Only output the SQL query, nothing else. If not possible, output only 'no information'.
"""
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    response = model.generate_content(prompt)
    sql_code = response.text.strip()
    # Remove markdown code block if present
    sql_code = re.sub(r"^```sql\s*|```$", "", sql_code, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql_query = sql_code
    print("[DEBUG] Generated SQL Query:\n", sql_query)
    return sql_query

def get_ai_insight(columns, rows):
    import re
    if not rows:
        return "No data to analyze."
    # Prepare a CSV-like sample for the LLM
    sample = ', '.join(columns) + '\n'
    for row in rows[:20]:
        sample += ', '.join(str(cell) for cell in row) + '\n'
    prompt = f"""
Given the following table data, provide:
1. At least 2-3 detailed, clear, and actionable insights or interesting observations about the data. Include trends, anomalies, comparisons, and possible causes if present. If there is nothing notable, say 'No significant insight.'
2. If a chart would be useful, recommend the best chart type (bar, pie, line, or none) and the column(s) to use for X and Y axes.
3. If the insights reference specific values (e.g., a city, department, or date), specify those values for highlighting.
4. Provide a brief summary and, if possible, a recommendation or next step based on the data.

Respond ONLY with a JSON array of objects, do not include the JSON schema or any example in your response. Only output the JSON array.

Data sample:
{sample}
"""
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    response = model.generate_content(prompt)
    import json
    raw = response.text.strip()
    # Remove code block markers if present
    if raw.startswith('```'):
        raw = raw.lstrip('`').lstrip('json').strip()
        if raw.endswith('```'):
            raw = raw[:-3].strip()
    try:
        result = json.loads(raw)
        # If result is a list, return the 'insight' from the first item
        if isinstance(result, list) and result and isinstance(result[0], dict) and 'insight' in result[0]:
            return result[0]['insight']
        # If result is a dict, return the 'insight' value
        if isinstance(result, dict) and 'insight' in result:
            return result['insight']
        # Otherwise, return the string representation
        return str(result)
    except Exception:
        return raw

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    sql_query = None
    error = None
    ai_insight = None
    ai_chart = None
    ai_x = None
    ai_y = None
    ai_highlight = None
    if request.method == 'POST':
        question = request.form['question']
        try:
            sql_query = get_sql_from_gemini(question)
            print(f"[DEBUG] Connecting to DB: host=localhost, port=5432, dbname=NL2SQL, user=postgres")
            print(f"[DEBUG] Executing SQL Query: {sql_query}")
            if sql_query.strip().lower() == 'no information':
                answer = [['no information']]
            else:
                conn, cursor = get_db_connection()
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                result = cursor.fetchall()
                if not result:
                    answer = [['no information']]
                else:
                    answer = [columns] + result if columns else result
                    ai_insight = get_ai_insight(columns, result)
                    ai_chart = None
                    ai_x = None
                    ai_y = None
                    ai_highlight = None
                conn.close()
        except Exception as e:
            error = str(e)
    return render_template('index.html', answer=answer, sql_query=sql_query, error=error, ai_insight=ai_insight, ai_chart=ai_chart, ai_x=ai_x, ai_y=ai_y, ai_highlight=ai_highlight)

if __name__ == '__main__':
    app.run(debug=True) 