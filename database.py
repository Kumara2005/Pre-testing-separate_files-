import sqlite3
from datetime import datetime
import pandas as pd
import os

class ResultsDB:
    """Handles the storage and retrieval of paper correction results."""
    
    def __init__(self, db_path="assessment_results.db"):
        """Initialize the database connection and ensure the schema exists."""
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Creates the results table with Roll Number and Teacher Calibration columns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Schema includes specific columns for AI vs Teacher comparison
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT,
                    roll_no TEXT,
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

    def insert_result(self, name, roll, subject, ai_score, teacher_score, max_m, grade):
        """
        Saves a finalized correction record.
        Calculates score_variance as (Teacher Score - AI Score).
        """
        # Ensure values are float for mathematical variance calculation
        ai_val = float(ai_score)
        teacher_val = float(teacher_score)
        variance = teacher_val - ai_val
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO results (
                    student_name, roll_no, subject, ai_score, teacher_score, 
                    score_variance, max_score, grade, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, roll, subject, ai_val, teacher_val, 
                variance, float(max_m), grade, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()

    def get_all_results_df(self) -> pd.DataFrame:
        """Retrieves all stored records as a pandas DataFrame for the Streamlit dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            # Order by ID descending to show the most recent evaluations at the top
            query = "SELECT * FROM results ORDER BY id DESC"
            return pd.read_sql_query(query, conn)

    def clear_database(self):
        """Optional helper to reset the record history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM results")
            conn.commit()