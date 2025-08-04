import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        sepal_length float8,
        sepal_width float8,
        petal_length float8,
        petal_width float8,
        target int
    );"""
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user=os.getenv("USER"), 
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT"),
        database=os.getenv("DATABASE"),
    )
    create_table(db_connect)