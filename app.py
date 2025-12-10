import psycopg2
import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from flask_cors import CORS
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np  
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API key
if not GEMINI_API_KEY:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

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
    model = genai.GenerativeModel('models/gemini-2.5-flash')
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
        return []
    # Prepare a CSV-like sample for the LLM
    sample = ', '.join(columns) + '\n'
    for row in rows[:20]:
        sample += ', '.join(str(cell) for cell in row) + '\n'
    prompt = f"""
Given the following table data, provide a JSON array of up to 5 insights. For each insight, include:
- type: (trend, anomaly, correlation, outlier, summary, or recommendation)
- description: A clear, concise insight or observation.
- recommendation: (if applicable) An actionable suggestion based on the insight.
- confidence: (optional) How confident you are in this insight (high/medium/low).

Respond ONLY with a JSON array of objects, no extra text or markdown.

Data sample:
{sample}
"""
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(prompt)
    import json
    raw = response.text.strip()
    # Remove code block markers if present
    if raw.startswith('```'):
        raw = raw.lstrip('`').lstrip('json').strip()
        if raw.endswith('```'):
            raw = raw[:-3].strip()
    # Extract JSON array
    import re
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not json_match:
        return [{"type": "summary", "description": raw, "recommendation": None, "confidence": "low"}]
    try:
        result = json.loads(json_match.group(0))
        if isinstance(result, list):
            return result
        else:
            return [{"type": "summary", "description": str(result), "recommendation": None, "confidence": "low"}]
    except Exception:
        return [{"type": "summary", "description": raw, "recommendation": None, "confidence": "low"}]

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(v) for v in obj]
    return obj

def get_table_columns(table):
    conn, cursor = get_db_connection()
    try:
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
        columns = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    return columns

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
                    ai_insight_list = get_ai_insight(columns, result)
                    ai_insight = '\n'.join(f"[{ins.get('type','')}] {ins.get('description','')}" + (f" Recommendation: {ins['recommendation']}" if ins.get('recommendation') else '') for ins in ai_insight_list)
                    ai_chart = None
                    ai_x = None
                    ai_y = None
                    ai_highlight = None
                conn.close()
        except Exception as e:
            error = str(e)
    return render_template('index.html', answer=answer, sql_query=sql_query, error=error, ai_insight=ai_insight, ai_chart=ai_chart, ai_x=ai_x, ai_y=ai_y, ai_highlight=ai_highlight)

@app.route('/predict', methods=['POST'])
def predict():
    import json
    import re
    data = request.get_json()
    question = data.get('question', '')
    # Use Gemini to interpret the question and select prediction type
    gemini_prompt = f"""
You are an expert data scientist. Given the following question, decide:
1. What type of prediction is required (time series forecast, regression, classification)?
2. Which table and columns from the schema are relevant?
3. What is the target variable to predict?
4. What is the prediction horizon (e.g., next 6 months)?
5. If columns from multiple tables are needed, specify the required JOINs in a 'joins' key in the JSON output. JOINs should be in the format 'table1.col1 = table2.col2'.
6. For time series, regression, or classification, the target must be a real column in the table, not an aggregate or invented column. For aggregate predictions, specify the aggregation and base column using a valid SQL aggregate function and provide an alias (e.g., COUNT(customer_id) AS count_of_new_customer_registrations).
7. Output a JSON with keys: type, table, features, target, joins (optional), horizon, and a brief explanation.

Schema:
{SCHEMA}

Question:
{question}

Respond ONLY with a valid JSON object, no markdown, no explanation, no extra text. Example:
{{"type": "regression", "table": "orders", "features": ["orders.order_date", "customers.city"], "target": "orders.total_amount", "joins": ["orders.customer_id = customers.customer_id"], "horizon": null, "explanation": "Predict total_amount using order date and customer city."}}
{{"type": "aggregate", "table": "customers", "features": ["registration_date"], "target": "COUNT(customer_id) AS count_of_new_customer_registrations", "joins": [], "horizon": 6, "explanation": "Forecast the count of new customer registrations for the next 6 months."}}
"""
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = gemini_model.generate_content(gemini_prompt)
    # Robustly extract the first JSON object from the response
    raw = response.text.strip()
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return {"error": "Could not interpret the question. (No JSON found)"}, 400
    try:
        parsed = json.loads(json_match.group(0))
    except Exception:
        return {"error": "Could not interpret the question. (Invalid JSON)"}, 400
    # Debug: print parsed Gemini response
    print('[DEBUG] Gemini parsed response:', parsed)
    pred_type = parsed.get('type', '').lower()
    table = parsed.get('table')
    features = parsed.get('features', [])
    target = parsed.get('target')
    horizon = parsed.get('horizon', None)
    explanation = parsed.get('explanation', '')
    # Get columns for the table before building the Gemini prompt
    table_columns = get_table_columns(table)
    table_columns_str = ', '.join(table_columns)
    gemini_prompt = f"""
You are an expert data scientist. Given the following question, decide:
1. What type of prediction is required (time series forecast, regression, classification)?
2. Which table and columns from the schema are relevant?
3. What is the target variable to predict?
4. What is the prediction horizon (e.g., next 6 months)?
5. If columns from multiple tables are needed, specify the required JOINs in a 'joins' key in the JSON output. JOINs should be in the format 'table1.col1 = table2.col2'.
6. For time series, regression, or classification, the target must be a real column in the table, not an aggregate or invented column. For aggregate predictions, specify the aggregation and base column using a valid SQL aggregate function and provide an alias (e.g., COUNT(customer_id) AS count_of_new_customer_registrations).
7. Output a JSON with keys: type, table, features, target, joins (optional), horizon, and a brief explanation.

Schema:
{SCHEMA}

For the table '{table}', the available columns are: {table_columns_str}. You may use columns from other tables if needed, but you must specify the JOINs required in the 'joins' key. Do not invent or create new columns. Only use columns that exist in the table(s) listed above, or valid aggregates over those columns (e.g., COUNT(customer_id) as count_of_new_customers).

Question:
{question}

Respond ONLY with a valid JSON object, no markdown, no explanation, no extra text. Example:
{{"type": "regression", "table": "orders", "features": ["orders.order_date", "customers.city"], "target": "orders.total_amount", "joins": ["orders.customer_id = customers.customer_id"], "horizon": null, "explanation": "Predict total_amount using order date and customer city."}}
{{"type": "aggregate", "table": "customers", "features": ["registration_date"], "target": "COUNT(customer_id) AS count_of_new_customer_registrations", "joins": [], "horizon": 6, "explanation": "Forecast the count of new customer registrations for the next 6 months."}}
"""
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = gemini_model.generate_content(gemini_prompt)
    # Robustly extract the first JSON object from the response
    raw = response.text.strip()
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return {"error": "Could not interpret the question. (No JSON found)"}, 400
    try:
        parsed = json.loads(json_match.group(0))
    except Exception:
        return {"error": "Could not interpret the question. (Invalid JSON)"}, 400
    print('[DEBUG] Gemini parsed response:', parsed)
    pred_type = parsed.get('type', '').lower()
    table = parsed.get('table')
    features = parsed.get('features', [])
    target = parsed.get('target')
    joins = parsed.get('joins', [])
    horizon = parsed.get('horizon', None)
    explanation = parsed.get('explanation', '')
    # Build SQL with JOINs if needed
    base_table = table
    sql = f"SELECT {', '.join(set(features + [target]))} FROM {base_table}"
    join_clauses = []
    for join in joins:
        # Parse join string like 'orders.customer_id = customers.customer_id'
        left, right = join.split('=')
        left_table = left.strip().split('.')[0]
        right_table = right.strip().split('.')[0]
        if left_table != base_table:
            join_clauses.append(f"JOIN {left_table} ON {join}")
        elif right_table != base_table:
            join_clauses.append(f"JOIN {right_table} ON {join}")
        else:
            join_clauses.append(f"JOIN {right_table} ON {join}")
    if join_clauses:
        sql += ' ' + ' '.join(join_clauses)
    # Add GROUP BY if aggregates and non-aggregates are mixed
    import re
    aggregate_pattern = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX)\b', re.IGNORECASE)
    aggregates = [col for col in features + [target] if aggregate_pattern.search(col)]
    non_aggregates = [col for col in features + [target] if not aggregate_pattern.search(col)]
    if aggregates and non_aggregates:
        sql += f' GROUP BY {", ".join(non_aggregates)}'
    # Parse aggregate target (e.g., COUNT(customer_id) AS count_of_new_customer_registrations)
    agg_target_match = re.match(r'^(COUNT|SUM|AVG|MIN|MAX)\((.+)\)\s+AS\s+([\w_]+)$', target.strip(), re.IGNORECASE)
    if agg_target_match:
        agg_func = agg_target_match.group(1).upper()
        base_col = agg_target_match.group(2).strip()
        alias = agg_target_match.group(3).strip()
        # Validate base_col exists
        if base_col.split('.')[-1] not in table_columns:
            return {"error": f"Aggregate base column '{base_col}' does not exist in the table(s): {table_columns}."}, 400
        sql_target = f"{agg_func}({base_col}) AS {alias}"
        sql_features = ', '.join(features)
        sql = f"SELECT {sql_features}, {sql_target} FROM {base_table}"
        if join_clauses:
            sql += ' ' + ' '.join(join_clauses)
        # Add GROUP BY for features
        if features:
            sql += f' GROUP BY {sql_features}'
        # Fetch data
        conn, cursor = get_db_connection()
        try:
            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=features + [alias])
        except Exception as db_exc:
            return {"error": f"Failed to fetch data with aggregate target. SQL: {sql}. Error: {str(db_exc)}"}, 400
        finally:
            conn.close()
        # Only allow predictions if alias is in df
        if alias not in df.columns:
            return {"error": f"Aggregate alias '{alias}' not found in result columns: {list(df.columns)}."}, 400
        # Continue with prediction using alias as target
        target = alias
    # Fetch relevant data from the table (sample up to 100 rows)
    conn, cursor = get_db_connection()
    try:
        cursor.execute(f"SELECT * FROM {table} LIMIT 100")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except Exception as db_exc:
        return {"error": f"Failed to fetch data from table '{table}'. Error: {str(db_exc)}"}, 400
    finally:
        conn.close()
    # Prepare CSV-like sample for Gemini
    sample = ', '.join(columns) + '\n'
    for row in rows:
        sample += ', '.join(str(cell) for cell in row) + '\n'
    # Prompt Gemini to answer the question and provide chart data and chart type
    direct_prompt = f"""
Given the following table data, answer the user's predictive question.
- Provide a summary answer as 'summary'.
- If a chart is appropriate, also return:
  - 'chart_type': "bar", "pie", or "line"
  - 'chart_data': a JSON array of data points for plotting (e.g., [{{"label": "...", "value": ...}}, ...])
- Respond with a JSON object: {{"summary": "...", "chart_type": "...", "chart_data": [...]}}

Data sample:
{sample}

Question:
{question}

Answer:
"""
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = gemini_model.generate_content(direct_prompt)
    import json
    raw = response.text.strip()
    # Try to parse as JSON object
    try:
        # Remove code block markers if present
        if raw.startswith('```'):
            raw = raw.lstrip('`').lstrip('json').strip()
            if raw.endswith('```'):
                raw = raw[:-3].strip()
        result = json.loads(raw)
        summary = result.get('summary', '')
        chart_data = result.get('chart_data', [])
        chart_type = result.get('chart_type', 'line')
    except Exception:
        summary = raw
        chart_data = []
        chart_type = 'line'
    return {"summary": summary, "chart_data": chart_data, "chart_type": chart_type}

if __name__ == '__main__':
    app.run(debug=True) 
