from elasticsearch import Elasticsearch
import json
import numpy as np
import logging
from openai import OpenAI
import os
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import time


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RagQuery():

    def __init__(
            self,
            llm_model: str='gemma2',
            eval_model: str='gemma2',
            index_name: str='manual_index',
            embedding_model_name: str='multi-qa-distilbert-cos-v1',
            search_type: str='knn',
            num_es_results: int=5,
            num_es_candidates: int=50,
            host: str='host.docker.internal',
            vehicle_name: str='Toyota Tacoma 2020',
            similarity_threshold: float=0.5,
            prompt_str: str='''\
Answer the QUESTION soley based on the CONTEXT from the user manual. Your response must include the relevant page numbers from the CONTEXT.'''
    ):
    
        self.llm_model=llm_model

        if host is None:
            host = os.getenv('HOST')
        if host is None:
            host = 'host.docker.internal'
        self.host = host

        self.eval_model = eval_model
        if embedding_model_name is None:
            self.embedding_model_name = os.getenv('SENTENCE_TRANSFORMER_MODEL_NAME')
        else:
            self.embedding_model_name=embedding_model_name

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.index_name=index_name
        self.search_type=search_type.lower().strip()
        self.num_es_results=num_es_results
        self.num_es_candidates=num_es_candidates
        self.vehicle_name=vehicle_name
        self.similarity_threshold = similarity_threshold
        self.prompt_str=prompt_str


    def rag_output_get_page_nums(self, rag_output: str):
        """Extract page numbers from RAG output. Return as a list.
        
        rag_output: Outout of rag function.
        """

        #pat = re.compile(r'(?<=page)([\W\dand]+)', flags=re.I)
        pat = re.compile(r'(page) (number)?s?\W*([\d\W]+)', flags=re.I)
        page_str = re.findall(pattern=pat, string=rag_output)
        if len(page_str) == 0 or page_str is None:
            return []
        
        pages = []

        for ps in page_str:
            for p in ps:
                p_txt = re.sub(pattern=r'[^\d]+', string=p, repl=',')
                pages.extend([i for i in p_txt.split(',') if i != ''])

        return list(set(pages))


    def evaluate_relevance(self, question, answer):
        logging.debug('Running evaluate_relevance')
        prompt_template = """
        You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
        Your task is to analyze the relevance of the generated answer to the given question.
        Based on the relevance of the generated answer, you will classify it
        as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

        Here is the data for evaluation:

        Question: {question}
        Generated Answer: {answer}

        Please analyze the content and context of the generated answer in relation to the question
        and provide your evaluation in parsable JSON without using code blocks:

        {{
        "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
        "Explanation": "[Provide a brief explanation for your evaluation]"
        }}
        """.strip()

        prompt = prompt_template.format(question=question, answer=answer)
        evaluation, tokens, _ = self.llm(prompt=prompt, model=self.eval_model)
        
        try:
            json_eval = json.loads(evaluation)
            return json_eval['Relevance'], json_eval['Explanation'], tokens
        except json.JSONDecodeError:
            return "UNKNOWN", "Failed to parse evaluation", tokens


    def calculate_openai_cost(self, tokens, model):
        logging.debug('Running calculate_openai_cost')
        openai_cost = 0
        if model == 'openai/gpt-3.5-turbo':
            openai_cost = (tokens.get('prompt_tokens', 0) * 0.0015 + tokens.get('completion_tokens', 0) * 0.002) / 1000
        elif model in ['openai/gpt-4o', 'openai/gpt-4o-mini']:
            openai_cost = (tokens.get('prompt_tokens', 0) * 0.03 + tokens.get('completion_tokens', 0) * 0.06) / 1000

        return openai_cost


    def make_es_query(
            self,
            user_question: str
            ):
        """From LLM Zoomcamp module. - Takes question as string and embedding model, return dict for Elasticsearch query."""

        logging.debug('Running make_es_query')
        es_client = Elasticsearch(f'http://{self.host}:9200') 

        if self.search_type == 'knn':
            question = self.embedding_model.encode(user_question)
            knn = {
                "field": "embedding",
                "query_vector": question,
                "k": self.num_es_results,
                "num_candidates": self.num_es_candidates,
                "similarity": self.similarity_threshold,
                "filter": {
                    "term": {
                        "vehicle": self.vehicle_name
                    }
        }
            }
            #res = es_client.search(
            #    index=self.index_name, knn=knn,
            #    source=['vehicle', 'page_ind', 'text'])
            res = es_client.search(
                index=self.index_name,
                knn=knn,
                size=self.num_es_results,
                _source=['vehicle', 'page_ind', 'text'])
            
        elif self.search_type == 'hybrid':
            question = self.embedding_model.encode(user_question)
            knn = {
                "field": "embedding",
                "query_vector": question,
                "k": self.num_es_results,
                "num_candidates": self.num_es_candidates,
                "similarity": self.similarity_threshold,
                "filter": {
                    "term": {
                        "vehicle": self.vehicle_name
                    }
                }
            }
            text_query= {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": user_question,
                                    "fields": ["text"],
                                    "type": "best_fields",
                                }
                            },
                            "filter": {
                                "term": {
                                "vehicle": self.vehicle_name
                                }
                            }   
                        }
            }
            
            res = es_client.search(
                index=self.index_name,
                query=text_query,
                knn=knn,
                size=self.num_es_results,
                _source=['vehicle', 'page_ind', 'text'])
        # Text only search
        else:
            search_query = {
                    "size": self.num_es_results,
                    "query": {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": user_question,
                                    "fields": ["text"],
                                    "type": "best_fields",
                                }
                            },
                            "filter": {
                                "term": {
                                "vehicle": self.vehicle_name
                                }
                            }   
                        }
                    },
                }

            res = es_client.search(index=self.index_name, body=search_query)
        
        es_client.transport.close()

        return [hit["_source"] for hit in res["hits"]["hits"]]


    def user_query_to_es(
        self,
        input_text: str):
        """Given an input string, embed the text and perform elasticearch search request."""

        logging.debug('Running user_query_to_es')
        results = self.make_es_query(user_question=input_text)
        
        return results


    def build_prompt(self,
            query: str,
            search_results: list[dict],
            prompt_str: str=None):
        """Adapted from LLM Zoomcamp Module 2. Format a user question along with Elasticsearch results as a LLM prompt.
        
        query: User question.
        search_results. Top N results from Elastic instance based on the user question.
        prompt_str: Base of the prompt to pass to a LLM, along with query and search_results.
        """
        logging.debug('Running build prompt')

    #Answer the QUESTION based on the CONTEXT from the user manual. 
    #If the provided CONTEXT does not provide the information needed to answer the QUESTION, then just respond with the page numbers from the CONTEXT
    #and tell the user to look at those page numbers.

        if prompt_str is None or prompt_str == '':
            prompt_str = self.prompt_str
            
        prompt_template = prompt_str + """

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

        context = ""
        
        for doc in search_results:
            context = context + f"Page: {doc['page_ind']}\nText: {doc['text']}\n\n"
        
        prompt = prompt_template.format(question=query, context=context).strip()
        return prompt


    def get_llm_client(self, model: str):
        """Build OpenAI or OLLAMA cient through OpenAI library
        
        model: string name of model. OpenAI vs OLLAMA selected based on model name.
        """
        logging.debug('Running get_llm_client')

        if 'openai' in model.lower() or 'gpt' in model.lower():
            client = OpenAI(
                organization=os.getenv('OPENAI_ORG'),
                project=os.getenv('OPENAI_PROJECT'),
                api_key=os.getenv('OPENAI_API_KEY'),
            )
        else:
            client = OpenAI(
            base_url=f'http://{self.host}:11434/v1/',
            api_key=os.getenv('OLLAMA_API_KEY'),
            )
        
        return client
        

    def llm(self, prompt, model: str='gemma2:2b'):
        """Directly from LLM Zoomcamp Module 2 and 4. Sends request to ollama instance."""

        logging.debug('Running llm method')
        client = self.get_llm_client(model=model)

        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        tokens = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        answer = response.choices[0].message.content

        end_time = time.time()
        response_time = end_time - start_time
        client.close()
        
        return answer, tokens, response_time


    def user_query_rewrite(self, user_question: str, model: str):
        """Handles questions asking for location of text in the manual by getting keywords from llm to search on elasticsearc."""

        rewriting_prompt = '''If the QUESTION below is in the form of "what page number...?" \
or "Where in the manual is...? then respond with keywords from the QUESTION. \
Otherwise respond with "none". QUESTION: \n\n'''
        logging.debug('Running user_query_rewrite method')
        query_rewrite, tokens, _ = self.llm(prompt=rewriting_prompt + user_question, model=model)

        if query_rewrite.strip().lower() == 'none':
            return user_question, tokens
        else:
            return query_rewrite, tokens
    
    
    def rag(
            self,
            query: str,
            evaluate: bool=False,
            prompt_str: str=None,
            user_query_rewrite: bool=False):
        """Modified from LLM Zoomcamp Module 2.
        
        Gets user input, retrieves best results from source data using elasticsearch,
        and passes both to LLM for answering the user question using the context from the source data.
        """

        logging.debug('Running rag method')

        rewritten_question = None
        rewrite_tokens = {}
        if user_query_rewrite:
            rewritten_question, rewrite_tokens = self.user_query_rewrite(user_question=query, model=self.eval_model)
            if rewritten_question == '' or rewritten_question is None:
                search_results = self.user_query_to_es(input_text=query)
            else:
                search_results = self.user_query_to_es(input_text=rewritten_question)
        else:
            search_results = self.user_query_to_es(input_text=query)
            
        if prompt_str is None or prompt_str == '':
            prompt = self.build_prompt(query=query, search_results=search_results)
        else:
            prompt = self.build_prompt(query=query, search_results=search_results, prompt_str=prompt_str)

        tokens = {}
        answer, tokens, response_time = self.llm(prompt, model=self.llm_model)

        relevance = None
        explanation = None
        eval_tokens = {}
        openai_cost = None
        eval_openai_cost = None
        rewrite_openai_cost = None

        #pages = None
        openai_cost = self.calculate_openai_cost(tokens=tokens, model=self.llm_model)
        rewrite_openai_cost = self.calculate_openai_cost(tokens=rewrite_tokens, model=self.eval_model)
        if evaluate:
            #pages = self.rag_output_get_page_nums(rag_output=answer)
            relevance, explanation, eval_tokens = self.evaluate_relevance(
                query, answer)
            eval_openai_cost = self.calculate_openai_cost(tokens=eval_tokens, model=self.eval_model)


        return {
            'answer': answer,
            'response_time': response_time,
            'relevance': relevance,
            'relevance_explanation': explanation,
            'model_used': self.llm_model,
            'question': query,
            'rewritten_question': rewritten_question,

            'prompt_tokens': tokens.get('prompt_tokens', None),
            'completion_tokens': tokens.get('completion_tokens', None),
            'total_tokens': tokens.get('total_tokens', None),

            'eval_prompt_tokens': eval_tokens.get('prompt_tokens', None),
            'eval_completion_tokens': eval_tokens.get('completion_tokens', None),
            'eval_total_tokens': eval_tokens.get('total_tokens', None),

            'rewrite_prompt_tokens': rewrite_tokens.get('prompt_tokens', None),
            'rewrite_completion_tokens': rewrite_tokens.get('completion_tokens', None),
            'rewrite_total_tokens': rewrite_tokens.get('total_tokens', None),
        
            'openai_cost': openai_cost,
            'eval_openai_cost': eval_openai_cost,
            'rewrite_openai_cost': rewrite_openai_cost,

            'pages': [i.get('page_ind', None) for i in search_results if i.get('page_ind', None)],
            'search_results': search_results
        }

    

if __name__ == '__main__':
    CD = Path(os.path.dirname(__file__))
    
    with open(Path('/rag_flask/data') / 'config.json', 'r') as f:
        config = json.load(f)

    rq = RagQuery(
            llm_model='gemma2',
            eval_model=config['eval_model'],
            index_name=config['index_name'],
            embedding_model_name=config['sentence_transformer_model_name'],
            search_type=config['search_type'],
            num_es_results=config['num_es_results'],
            num_es_candidates=config['num_es_candidates'],
            vehicle_name='Toyota Tacoma 2020',
            similarity_threshold=config['es_knn_similarity_threshold'],
            prompt_str=config['prompt']
        )

    #try:
    if config.get('rewrite_user_query', 'false').lower().strip() == 'true':
        config['rewrite_user_query'] = True
    else:
        config['rewrite_user_query'] = False
    
    if config.get('evaluate_response', 'false').lower().strip() == 'true':
        config['evaluate_response'] = True
    else:
        config['evaluate_response'] = False

    print('rewrite_user_query:', config['rewrite_user_query'])
    print('evaluate_response:', config['evaluate_response'])
    llm_response = rq.rag(
        query='Where is the tire jack kept?',
        evaluate=config['evaluate_response'],
        user_query_rewrite=config['rewrite_user_query'],
        )
    
    for k, v in llm_response.items():
        print(k, v)