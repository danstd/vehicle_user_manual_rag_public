from elasticsearch import Elasticsearch
import logging
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
import re
import requests
from sentence_transformers import SentenceTransformer

import db

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataDownloadPreprocess():
    """Class for download and preprocess of source PDF file."""
    def __init__(self, host: str=None, output_path: Path=None):
        #CD = globals()['_dh'][0]
        CD = Path(__file__).parent.resolve()
        with open(CD / 'data' / 'config.json', 'r') as f:
            self.config = json.load(f)
            self.vehicles = self.config['vehicles']

        if output_path is None:
            self.output_path = CD / 'data'
        else:
            self.output_path = output_path
        if (host is None) or (host == ''):
            self.host = os.getenv('host')
        else:
            self.host=host

    def get_pdf(self, url: str=None, output_file_name: str='output.pdf'):
        """Retrieve and save pdf from source url
        
        url: Location of pdf to download.
        name_prefix: Prefix for generic 'output.pdf' name
        """
        # source URL
        if (url is None) or (url == ''):
            url = os.getenv('SOURCE_URL')

        # Request and saving data.
        r = requests.get(url)

        print('writing pdf to:', str(self.output_path / output_file_name))
        with open(self.output_path / output_file_name, 'wb') as f:
            f.write(r.content)

        return output_file_name


    def process_pdf(
            self,
            input_file_name: str='output.pdf',
            output_file_name: str='output.json',
            vehicle_name: str=''):
        """Reading PDF text with PdfReader and simple clean-up.
        
        file_name: name of pdf file to read
        output_file_name: name of output json file.
        """
        
        num_newline = re.compile(r'(.(?=\d[A-Z]))')
        case_newline = re.compile(r'([a-z][A-Z])')

        f = PdfReader(self.output_path / input_file_name,
)

        # Process each page in the pdf.
        page_output = list()
        page_counter = 1
        for page in f.pages:
            pdf_page = page.extract_text()

            pdf_page = pdf_page.replace('\n', ' ')

            newline_inserts = list()
            for mtch in re.finditer(string=pdf_page, pattern=num_newline):
                newline_inserts.append([mtch.span()[0], 'num'])

            for mtch in re.finditer(string=pdf_page, pattern=case_newline):
                newline_inserts.append([mtch.span()[0], 'case'])

            newline_inserts.sort(key=lambda x: x[0], reverse=True)

            pdf_page = list(pdf_page)
            for mtch in newline_inserts:
                if mtch[1] == 'case':
                    pdf_page.insert(mtch[0]+1, '\n')
                else:
                    pdf_page.insert(mtch[0]+2, ' ')
                    pdf_page.insert(mtch[0]+1, '\n')

            pdf_page = ''.join(pdf_page)
            page_output.append({'vehicle': vehicle_name, 'page_ind': page_counter, 'text': pdf_page})
            page_counter += 1

        # Save output as JSON
        page_output = json.dumps(page_output, indent='  ')


        with open(self.output_path / output_file_name, 'w', encoding='utf16') as f:
            f.write(page_output)


    def embed_text(
            self,
            json_file_name: str='output.json',
            model_name: str='',
            output_file_name: str='output_embedding'
            ):
        """Calculate text embeddings:
        
        json_file_name: Name of json file to import.
        model name: sentence embedding model to import.
        """

        if model_name == '' or model_name is None:
            model_name = os.getenv('MODEL_NAME')
            if model_name is None:
                model_name='multi-qa-distilbert-cos-v1'

        print('Using model name:,', model_name)
        embedding_model = SentenceTransformer(model_name)
        with open(self.output_path / json_file_name, 'r', encoding='utf16') as f:
            pages = json.loads(f.read())

        for page in pages:
            page['embedding'] = embedding_model.encode(page['text'])

        pages = pd.json_normalize(pages) #['page_ind', 'text', 'embedding']

        output_name = self.output_path / output_file_name
        pages.to_parquet(output_name)
        print(f'output saved at: {output_name}')
        return None


    def elasticsearch_embed(
            self,
            input_file_name: str='output_embedding.parquet',
            index_name: str='manual_index',
            delete_create_index: bool=True):
        """Create elasticsearch embeddings.
        
        input_file_name: Output parquet file name from embed_text method.
        index_name: name of elasticsearch index
        delete_create_index: If true, delete and recreate index before inserting.
        """
        
        if index_name == '' or index_name is None:
            index_name = self.config['index_name']

        print('using index name:', index_name)

        df = pd.read_parquet(self.output_path / input_file_name)
        vec_len = df['embedding'].values[0].shape[0]

        es_client = Elasticsearch(f'http://{self.host}:9200') 

        index_settings = {
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0
            },
            'mappings': {
                'properties': {
                    'vehicle': {'type': 'keyword'},
                    'text': {'type': 'text'},
                    'page_ind': {'type': 'integer'},
                    'embedding': {'type': 'dense_vector', 'dims': vec_len, 'index': True, 'similarity': self.config['es_knn_similarity_measure']},
                }
            }
        }
        if delete_create_index:
            es_client.indices.delete(index=index_name, ignore_unavailable=True)
            es_client.indices.create(index=index_name, body=index_settings)

        doc_counter = 1
        for doc in df[['vehicle', 'page_ind', 'text', 'embedding']].to_dict('records'):
            es_client.index(index=index_name, document=doc)
            doc_counter += 1
        print(f'{doc_counter} records added')


def main():
    if os.getenv('host') is None:
        host = 'host.docker.internal'  # 'localhost' for running out of container.
    else:
        host = os.getenv('host')

    logger.debug(f"Using host: {host}")

    # run datapipeline for both model options, creating different elasticsearch indices.

    dp = DataDownloadPreprocess(host=host)

    for v_ind, v in enumerate(dp.vehicles):
        elastic_delete_create_index = v_ind == 0
        
        logger.info(f"Processing vehicle: {v['vehicle']}, elastic_delete_create_index: {elastic_delete_create_index}")

        logger.info('Downloading PDF')

        pdf_file_name = dp.get_pdf(url=v['url'], output_file_name=v['prefix'] + 'output.pdf')
        logger.info(f"PDF downloaded as: {pdf_file_name}")

        logger.info('Processing PDF')
        json_output_file_name=v['prefix'] + 'output.json'
        dp.process_pdf(
            input_file_name=pdf_file_name,
            output_file_name=json_output_file_name,
            vehicle_name=v['vehicle']
        )
        logger.info(f"PDF processed to json, saved to: {json_output_file_name}")

        logger.info('Embedding text')
        parquet_output_file_name=v['prefix'] + dp.config['transformer_model_prefix'] + 'output_embedding.parquet'
        dp.embed_text(
            model_name=dp.config['sentence_transformer_model_name'],
            json_file_name=json_output_file_name,
            output_file_name=parquet_output_file_name
        )
        logger.info(f"Embedded text along with data columns saved to: {parquet_output_file_name}")

        logger.info('Moving to Elasticsearch')
        dp.elasticsearch_embed(
            input_file_name=parquet_output_file_name,
            index_name=None,
            delete_create_index=elastic_delete_create_index
        )
        logger.info(f"Data moved to Elasticsearch index: {dp.config['transformer_model_prefix'] + 'manual_index'}")

    logger.info("Initializing database")
    db.init_db()
    logger.info("Database initialized")


if __name__ == '__main__':
    main()