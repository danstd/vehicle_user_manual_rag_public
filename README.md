# Vehicle User Manual RAG Question and Answer

### Problem Description and Project Overview
Car user manuals are frustrating to use even in the best of circumstances, and are highly difficult to use in stressful situations caused by car troubles where they are most likely to be needed. User manual indexes are often less than helpful, and the worthwhile information is often buried in frustrating warnings designed more to protect manufacturers from liability than provide vehicle information.

The aim of this project is to create an application to allow users to easily ask questions and quickly and succintly retrieve information from their vehicle's user manual, including simple responses about the best page numbers to look for. This is essential as often the relevant vehicle information is only given in vague diagrams that would likely not be parsed well enough to give instructions by image-to-text models.

The initial goal for this project was to create a working application for a single vehicle type, but enhancements have been made to include additional vehicles. As a working prototype, the 2020 Toyota Tacoma and 2020 Toyota Rav4 are included as vehicle search options.

### Technologies
 - *LLM*: OpenAI and Ollama are both usable with this application. Gemma2 and gpt-4o-mini are given as options in the [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file (see below), but other models may be added by updating the config.json file.
 - *Knowledge base*: Elasticsearch is used to store text as wel as vector embeddings for the user manual information.
 - *Monitoring*: This application does not have a front-end enabled for monitoring, but user interactions and user feedback are recorded and saved to a Postgres database. The [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file can be used to enable or disable online LLM evaluation of generated answers and user query rewriting.
- *Interface*: The user interface for this project is a Flask application. [Flask is a popular lightweight web application framework for Python](https://palletsprojects.com/projects/flask/).
- *Ingestion pipeline*: A python file is used to run the ingestion pipeline for this application - *data_pipeline.py*.

### Best Practices
- Hybrid search with elasticsearch may be used if set in the [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file. By default the best settings as determined by evaluation (see [evaluation/es_retrieval_evaluation.ipynb](https://github.com/danstd/vehicle_user_manual_rag/blob/main/evaluation/es_retrieval_evaluation.ipynb)) are already set in the [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file.
- User query rewriting may be used to handle user questions asking for the location of information in the user manuals; this is also controlled in the [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file.

- 
### Instructions

#### Quick start
 - Clone this repository: `git clone https://github.com/danstd/vehicle_user_manual_rag.git`
 - Move to new directory: `cd vehicle_user_manual_rag`
 - Create *.env* file. `cp .env_reference .env`
 - Add OpenAI keys to *.env* file if you would like to use OpenAI models.
 - Build and start docker containers: `docker-compose up -d`
 - Connect to Flask container: `docker exec -it u_manual_rag_flask sh` or `winpty docker exec -it u_manual_rag_flask sh`
    - Run data pipeline: `python data_pipeline.py`.
 - For Ollama models:
    - Connect to Ollama container: `docker exec -it u_manual_ollama sh` or `winpty docker exec -it u_manual_ollama sh`
    - Run `ollama pull gemma2`
 - Go to `localhost:5000` in your browser to access the application.

   
#### *.env File*
- This application can be run entirely locally through *docker-compose*. The file *.env_reference* contains a list of needed environment variables, with secret values removed. There must be a file name *.env* in the base directory (same directory as the *docker-compose.yml* file). This file can be created using the .env_reference file. Remove the comments and set variables, save as *.env*. Explanations for the environment variables are as follows:

##### Environment variable explanation:
    - POSTGRES_USER *set as desired for local database*
    - POSTGRES_PASS *set as desired for local database*
    - POSTGRES_DB *set as desired for local database*
    - HOST=host.docker.internal *host for communication between docker containers.*
    - OLLAMA_API_KEY *Set for local ollama instance. The key is required, but is ignored.*
    - OPENAI_API_KEY *Needed for using openai models - not necessary for the application otherwise.*
    - OPENAI_ORG *May be needed for using openai models - not necessary for the application otherwise.*
    - OPENAI_PROJECT *May be needed for using openai models - not necessary for the application otherwise.*
    - FLASK_APP=rag_flask *name of Flask application*
    - SECRET_KEY *Flask app secret key. Needed for creating session IDs. May be set as desired for local application.*

#### [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) File (rag_flask/data/config.json)
The config file contains necessary variables and data for running the Flask application. By default no changes need to be made for the application to function.

##### Config JSON file explanation
- *search_type*: Type of search to perform on Elasticsearch. Options are knn, hybrid, and text.
- *all_search_type_options*: Reference for search_type.
- *sentence_transformer_model_name*: Embedding model for elasticsearch queries, if *knn* or *hybrid* are used.
- *transformer_model_prefix*: Prefix for corresponding elastisearch index name that uses the model.
- *index_name*: Elasticsearch index corresponding to the embedding model.
- *num_es_results*: Number of elasticsearch results to return and pass to the LLM as context.
- *num_es_candidates*: Number of candidates for each elasticsearch batch.
- *es_knn_similarity_measure*: Similarity measure for knn and hybrid elasticsearch queries.
- *es_knn_similarity_threshold*: Minimum similarity theshold for knn and hybrid elasticsearch queries.
- *eval_model*: LLM for evaluating RAG output, and rewriting user queries.
- *rewrite_user_query*: Rewrite user queries if True.
- *evaluate_response*: Use a LLM for online evaluation.
- *model_options*: LLM options for RAG.
- *vehicles*: contains information for vehicles to ask questions about.
    - *vehicle*: Vehicle make, model, and year. Display name.
    - *url*: Accessible URL for the vehicle's user manual.
    - *prefix*: Prefix for vehicle-specific files, created and used in data processing by *data_pipeline.py*.
- *prompt*: Base prompt passed to the LLM along wih context to create the RAG output.

#### Docker-Compose
The application may be run by cloning this repository, setting the .env file, and running *docker-compose up*. Note that the application may have issues running with less than ~14 GB of RAM made available to Docker if you are using Ollama models.

After the docker containers are created and have started, connect to the *toyota_rag_flask* container through docker desktop or via command line using
`docker exec -it toyota_rag_flask sh`. Then, run the data pipeline script simply by running `python data_pipeline.py`. After that finishes running you should be all set!

#### Accessing the application.
After starting the Flask application will be available at *localhost:5000* in your browser.


### Directories and files:
- *rag_flask* is the main project directory.
    - *data_pipeline.py*: The data pipeline file. This can be run as-is to complete all project set-up.
    - *db.py*: Contains functions to set up and interact with the postgres database.
    - *dockerfile*: Dockerfile for flask application container (titled 'back-end' in *docker-compose.yaml*).
    - *rag_flask.py*: This is the Flask application file. Because there is only a single route, this is essentially a single-file application.
    - *rag_query.py*: This file contains the RagQuery class which handles all RAG functions- searching using elasticsearch (incuding text, vector, and hybrid search), interacting with local Ollama container or with the OpenAI API. This class contains methods for user query rewriting to handle user questions about the location of information in the user manual (e.g. 'What page is X on?'). Online evaluation based on DataTalks Club Zoomcamp module 4 is enabled, including recording cost and tokens of evaluation and user query rewriting if applicable. Most methods are loosley coupled to enable testing and evaluation of retrieval and offline RAG performance.
    - *requirements.txt* The python requirements file for the application.

 - *data*: This directory contains reference and downloaded data needed for the application. This directory stores downloaded pdf files, processed json files of the pdfs, parquet files from the json data including sentence-transformer embedded text, and the aforementioned [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file. Only the [config.json](https://github.com/danstd/vehicle_user_manual_rag/blob/main/rag_flask/data/config.json) file is needed to start the application. the rest are created by the *data_pipeline.py* run.
    <br>
    - *templates*: this directory contains the single jinja2 html template file for the Flask application, *user_query.html*.
    <br>

- *evaluation*: This directory stores notebooks for elasticsearch retrieval evaluation and RAG evaluation.
    - *es_retrieval_evaluation.ipynb*: Evaluation of elasticsearch retrieval. Evaluates text, KNN, and hybrid search. Because chunking is handled by user manual pages and multiple pages may contain relevant information, the evaluation includes modified hit rate and mean reciprocal rank measures, in addition to standard measures, that compare the presence and rank respectively of all manually-determined relevant pages in the ground truth dataset.

    - *rag_evaluation.ipynb*: Offline Answer-Question-Answer evaluation of a manual dataset of user manual questions on the Tacoma user manual. A LLM is used to evaluate the relevance of LLM answers compared to the ground-truth answers. This notebook is based on the offline_rag_evaluation notebook in the [DataTalks LLM Zoomcamp Module 4](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/offline-rag-evaluation.ipynb)

