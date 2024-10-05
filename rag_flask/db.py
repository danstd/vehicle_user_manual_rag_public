'''This file is from DataTalksclub LLM Zoomcamp Module 4
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/app/db.py
'''
import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo

tz = ZoneInfo('America/New_York')


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'host.docker.internal'),
        database=os.getenv('POSTGRES_DB', 'MANUAL_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )


def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS feedback')
            cur.execute('DROP TABLE IF EXISTS page_reference')
            cur.execute('DROP TABLE IF EXISTS conversation')

            cur.execute('''
                CREATE TABLE conversation (
                    conversation_id SERIAL PRIMARY KEY,
                    conversation_session_id TEXT NOT NULL,
                    vehicle TEXT NOT NULL,
                    question TEXT NOT NULL,
                    rewritten_question TEXT,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT,
                    relevance TEXT,
                    relevance_explanation TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    eval_prompt_tokens INTEGER,
                    eval_completion_tokens INTEGER,
                    eval_total_tokens INTEGER,
                    openai_cost FLOAT,
                    eval_openai_cost FLOAT,
                    rewrite_openai_cost FLOAT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            ''')
            cur.execute('''
                CREATE TABLE feedback (
                    feedback_id SERIAL PRIMARY KEY,
                    conversation_session_id TEXT NOT NULL,
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            ''')
            cur.execute('''
                CREATE TABLE page_reference (
                    page_reference_id SERIAL PRIMARY KEY,
                    vehicle TEXT NOT NULL,
                    conversation_session_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            ''')
        conn.commit()
    finally:
        conn.close()


def save_conversation(conversation_session_id, vehicle, answer_data, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO conversation 
                (conversation_session_id, vehicle, question, rewritten_question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, eval_openai_cost, rewrite_openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
            ''',
                (
                    conversation_session_id,
                    vehicle,
                    answer_data['question'],
                    answer_data['rewritten_question'],
                    answer_data['answer'],
                    answer_data['model_used'],
                    answer_data['response_time'],
                    answer_data['relevance'],
                    answer_data['relevance_explanation'],
                    answer_data['prompt_tokens'],
                    answer_data['completion_tokens'],
                    answer_data['total_tokens'],
                    answer_data['eval_prompt_tokens'],
                    answer_data['eval_completion_tokens'],
                    answer_data['eval_total_tokens'],
                    answer_data['openai_cost'],
                    answer_data['eval_openai_cost'],
                    answer_data['rewrite_openai_cost'],
                    timestamp,
                ),
            )
            page_data = [(conversation_session_id, vehicle, i, timestamp) for i in answer_data['pages']]
            page_data = [i for i in page_data if i]
            if len(page_data) > 0:
                cur.executemany(
                    '''
                    INSERT INTO page_reference 
                    (conversation_session_id, vehicle, page, timestamp)
                    VALUES (%s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
                ''',
                    page_data
                )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_session_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO feedback (conversation_session_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))',
                (conversation_session_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
    init_db()
