from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for, session, jsonify
import json
import logging
import os
from pathlib import Path
import psycopg2
import re
import uuid

import db
import rag_query


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

def user_question_to_rag(req_data: dict, config: dict):
    """Handles passing user question to rag_query.
    
    req_data: dict from form response.
    config: dictionary from config.json file
    """
    
    logging.info('Running user_question_to_rag')
    user_question = req_data['user_question']
    question_model = req_data['question_model']
    vehicle_option = req_data['question_vehicle']

    rq = rag_query.RagQuery(
        llm_model=question_model,
        eval_model=config['eval_model'],
        index_name=config['index_name'],
        embedding_model_name=config['sentence_transformer_model_name'],
        search_type=config['search_type'],
        num_es_results=config['num_es_results'],
        num_es_candidates=config['num_es_candidates'],
        vehicle_name=vehicle_option,
        similarity_threshold=config['es_knn_similarity_threshold'],
        prompt_str=config['prompt']
    )

    llm_response = rq.rag(
        query=user_question,
        evaluate=config['evaluate_response'],
        user_query_rewrite= config['rewrite_user_query']
        )
    logging.info(f'RAG query successful, response: {llm_response}')
    #except Exception as e:
        #llm_response = repr(e)
        #logging.error(f'Exception occurred during RAG query: {llm_response}')

    return llm_response


@app.route("/", methods=["GET", "POST"])
def user_query_page():
    with open(Path('/rag_flask/data') / 'config.json', 'r') as f:
        config = json.load(f)
    if config.get('rewrite_user_query', 'false').lower().strip() == 'true':
        config['rewrite_user_query'] = True
    else:
        config['rewrite_user_query'] = False
    
    if config.get('evaluate_response', 'false').lower().strip() == 'true':
        config['evaluate_response'] = True
    else:
        config['evaluate_response'] = False

    vehicle_options = [i['vehicle'] for i in config['vehicles']]
    model_options = config['model_options']
    
    if request.method == "GET":
        session['conversation_session_id'] = uuid.uuid4()
        logging.info('GET request received for user_query_page')
        return render_template('user_query.html', model_options=model_options, vehicle_options=vehicle_options)
    
    req_data = request.get_json()
    logging.info(f'POST request received with data: {req_data}')
    if 'user_question' in req_data:
        # Check if user_question is empty or only whitespaces
        if re.match(pattern=r'^\s*$', string=req_data['user_question']) is not None:
            logging.warning('User submitted an empty or whitespace-only question.')
        
            results = {'return_display': 'Please enter a valid question!',
                       'model_options': model_options, 'vehicle_options': vehicle_options}
            return jsonify(results)

        if req_data['question_model'] == 'dummy':
            logging.info('Dummy model request')
            llm_response = {'answer': 'dummy model for checking web actions'}
            results = {'return_display': llm_response['answer'], 'model_options': model_options, 'vehicle_options': vehicle_options}
            return jsonify(results)
        else:
            llm_response = user_question_to_rag(req_data, config)
    
        logging.info(f"Saving to conversation table. Conversation ID: {str(session['conversation_session_id'])}")
        db.save_conversation(
            conversation_session_id=str(session['conversation_session_id']),
            vehicle=req_data['question_vehicle'],
            answer_data=llm_response)
        
        results = {'return_display': llm_response['answer'], 'model_options': model_options, 'vehicle_options': vehicle_options}
        return jsonify(results)
    else:

        try:
            logging.info(f"Saving to feedback table. Conversation ID: {str(session['conversation_session_id'])}")
            db.save_feedback(
            conversation_session_id=str(session['conversation_session_id']),
            feedback=req_data['feedback']
            )
            feedback_msg = 'Your feedback has been recorded. Thank you!.'
        except psycopg2.errors.ForeignKeyViolation:
            logging.warning('Foreign key violation while saving to feedback table. Conversation ID:', session['conversation_session_id'])
            feedback_msg = 'Please use the application before submitting feedback. Thank you!.'

        
        return jsonify({'model_options': model_options, 'vehicle_options': vehicle_options, 'feedback_msg': feedback_msg})
    #return render_template('user_query.html', return_display=llm_response['answer'],
    #                       model_options=model_options)