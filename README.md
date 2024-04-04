# Generative AI RAG Project

## Description :

This is a project to make an interface to use RAG easily

## Installation :

To install this app on your local machine, follow these steps:

- Install python 3.10 or higher
- Install git if you don't have
- Open a terminal in your specified folder
- Clone the project with this command ``git clone https://github.com/HolzerSoahita/Generative_AI_RAG_Project.git``
- Navigate to the app directory with the command ``cd Generative_AI_RAG_Project``
- Create a virtual environment named **ragvenv** using the command: ``python -m venv ragvenv``
- Activate the **ragvenv** environment by running the following command:
  - On Windows: ``ragvenv\Scripts\activate.bat``
  - On Linux or Mac: ``source ragvenv/bin/activate``
- Install the requirements.txt file using the command: ``pip install -r requirements.txt``
- Run the app with the command: ``python main.py``. This will launch a web server on your local machine and click the link generated in the console to access the gradio app.
- To stop the server press CTRL+C   
**NB**: If you try to execute this code on Google Colab, the link generated doesn't work because it is blocked by Google Colab

## RAG Experience :

* This experience is released with the model ``TheBloke/Mistral-7B-Instruct-v0.2-GGUF``.
* We use size input 256 and overlap 20 on the chunking
* The folder of used dataset(one folder by use case) and the related questions can be found in this [link](dataset_example)
* The result of the RAG experience is sum up with this following table :

| Chunking Algorithm     | Number of documents | Type of documents | use case                     | Number of questions | Number of true predicted values |
| ---------------------- | ------------------- | ----------------- | ---------------------------- | ------------------- | ------------------------------- |
| Fixed sized chunking   | 3                   | pdf               | Scientific papers            | 10                  | 10                              | 
| Content-aware chunking | 3                   | docx, pptx        | slides and technical reports | 10                  | 10                              | 
| Recursive chunking     | 3                   | pdf               | Legal document Analysis      | 10                  | 10                              | 
| Semantic chunking      | 3                   | csv, excel, text  | personal docs                | 10                  | 10                              | 
| Agentic chunking       | 3                   | csv, excel, text  | personal docs                | 10                  | 10                              | 

**NB**: I recommend to choose a text generation model that can support chat format to have better result

## Usage :

The user documentation of this app can be found [here](user_documentation.pdf)

## Code documentation :

The main file is `main.py`.
In the folder [library](library) , you can find all the libraries needed for this app. These libraries are:

- `agentic_chunker.py` : File class to make the agentic chunking
- `chunking.py` : all the functions needed for the chunking and saving vector database
- `FileSaver.py` : File class for saving an html code to file format(html,pdf,docx)
- `rag.py` : File class to make the RAG(Retrieval Augmented Generation)

## Author :

SOAHITA Salvoldi Holzer
PGE 5 Student at Aivancity University Cachan Paris
