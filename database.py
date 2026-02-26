import sqlite3
from datetime import datetime
import pandas as pd
import os

class ResultsDB:
    def __init__(self, db_path="assessment_results.db"):
        self.db_path = db_path
        # This call must match the method name defined below
        self._create_table()

    def _create_table(self):
        """Creates the results table with calibration columns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT,
                    subject TEXT,
                    ai_score REAL,
                    teacher_score REAL,
                    score_variance REAL,
                    max_score REAL,
                    grade TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()

    def insert_result(self, name, subject, ai_score, teacher_score, max_m, grade):
        """Saves a record and calculates the variance between AI and Teacher."""
        variance = float(teacher_score) - float(ai_score)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO results (
                    student_name, subject, ai_score, teacher_score, 
                    score_variance, max_score, grade, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, subject, ai_score, teacher_score, variance, max_m, grade, 
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    def get_all_results_df(self):
        """Returns all records for the dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)