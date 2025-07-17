import psycopg2
from sqlalchemy import create_engine
import pandas as pd

# Connect to PostgreSQL database
engine = create_engine('postgresql+psycopg2://postgres:Sahith%40123@localhost:5432/NL2SQL')
conn = engine.connect()

# Load and execute schema
with open("employees_schema.sql", "r") as f:
    schema = f.read()
drop_tables = '''
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS projects;
'''
sql_script = drop_tables + schema
raw_conn = engine.raw_connection()
cursor = raw_conn.cursor()
cursor.execute(sql_script)
raw_conn.commit()
cursor.close()
raw_conn.close()

# Load and insert CSV data
df = pd.read_csv("employees_large.csv")
df.to_sql("employees", engine, if_exists="replace", index=False)

conn.close()

print("Database created and populated successfully.")
