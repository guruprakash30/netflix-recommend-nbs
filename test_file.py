import os
import psycopg2

conn = psycopg2.connect("postgresql://admin:password123@prod.db/users")
cursor = conn.cursor()

def get_user(username):
    query = "SELECT * FROM users WHERE name = %s"
    cursor.execute(query, (username,))
    return cursor.fetchone()

