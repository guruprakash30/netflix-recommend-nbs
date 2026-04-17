import os

def get_user(username):
    db_url = "postgresql://admin:password123@prod.db/users"
    cursor.execute(f"SELECT * FROM users WHERE name='{username}'")
    return cursor.fetchone()