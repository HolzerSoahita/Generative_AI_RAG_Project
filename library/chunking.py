from langchain_community.document_loaders import Docx2txtLoader,UnstructuredPowerPointLoader,TextLoader,PyPDFLoader,CSVLoader,UnstructuredMarkdownLoader,BSHTMLLoader
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter,Language,HTMLHeaderTextSplitter,MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .agentic_chunker import AgenticChunker
import pandas as pd
import json, os, re


def universal_loader(file_path):
    """
        Function to load each type of file for the RAG 
    """

    # Determine the file extension
    extension = (file_path.split('.')[-1]).lower()
    print(f"extension : {extension}")
    
    if extension == 'docx':
        print("type word")
        loader = Docx2txtLoader(file_path)
    elif extension == 'xlsx':
        print("type excel")
        doc_excel = pd.read_excel(file_path) 
        csv_file_path = os.path.splitext(os.path.basename(file_path))[0] + '.csv'
        doc_excel.to_csv(csv_file_path, index=False)
        loader = CSVLoader(file_path=csv_file_path)
    elif extension == 'pptx':
        print("type powerpoint")
        loader = UnstructuredPowerPointLoader(file_path,mode="elements")
    elif extension == 'pdf':
        print("type pdf")
        loader = PyPDFLoader(file_path)
    elif extension == 'txt':
        print("type text")
        loader = TextLoader(file_path)
    elif extension == 'csv':
        print("type csv")
        loader = CSVLoader(file_path)
    elif extension == 'md':
        print("type markdown")
        loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    elif extension == 'html':
        print("type html")
        loader = BSHTMLLoader(file_path)
#         loader = UnstructuredHTMLLoader(file_path)
    elif extension == 'json':
        print("type json")
        return json.loads(Path(file_path).read_text())
#        loader = JSONLoader(file_path=file_path,jq_schema='.')
    else:
        print("type unknown")
        raise Exception('Type unknown!')
        
    return loader.load()


def fixed_size_chunking(path,chunk_size,chunk_overlap):
    """
        Function to make fixed size chunking for one file
    """
    
    # Load data
    documents = universal_loader(path)
        
    # Define fixed size chunking
    text_splitter = CharacterTextSplitter(
         separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents with fixed size chunking
    text = text_splitter.split_documents(documents)
    return text

def recursive_chunking(path,chunk_size,chunk_overlap):
    """
        Function to make recursive chunking for one file
    """
    
    # Load data
    documents = universal_loader(path)
    
    # Define recursive chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents with recursive chunking
    text = text_splitter.split_documents(documents)
    
    return text

def content_aware_chunking(path, chunk_size, chunk_overlap):
    """
        This function performs content-aware chunking on a file based on its extension. 
    """
    
    # Determine the file extension
    extension = (path.split('.')[-1]).lower()
    print(f"extension : {extension}")
        
    Extension_CodeTextSplitter = ['cpp', 'go', 'java', 'kt', 'js', 'ts', 'php', 'proto', 'py', 'rst', 'rb', 'rs', 'scala', 'swift', 'md', 'tex', 'html', 'sol', 'cs', 'cob']

        
    if extension in Extension_CodeTextSplitter:
        # Load data
        with open(path) as file:
            data = file.read()
        print(f"Data : {type(data)}")
        
        # Chunking base on extension of file
        if extension == 'cpp':
            cpp_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            cpp_docs = cpp_splitter.create_documents([data])
            
            return cpp_docs
        elif extension == 'go':
            go_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.GO, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            go_docs = go_splitter.create_documents([data])
            
            return go_docs
        elif extension == 'java':
            java_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JAVA, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            java_docs = java_splitter.create_documents([data])
            
            return java_docs
        elif extension == 'kt':
            kt_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.KOTLIN, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            kt_docs = kt_splitter.create_documents([data])
            
            return kt_docs
        elif extension == 'js':
            js_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            js_docs = js_splitter.create_documents([data])
            
            return js_docs
        elif extension == 'ts':
            ts_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.TS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            ts_docs = ts_splitter.create_documents([data])
            
            return ts_docs
        elif extension == 'php':
            php_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PHP, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            php_docs = php_splitter.create_documents([data])
            
            return php_docs
        elif extension == 'proto':
            proto_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PROTO, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            proto_docs = proto_splitter.create_documents([data])
            
            return proto_docs
        elif extension == 'py':            
            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            python_docs = python_splitter.create_documents([data])
            
            return python_docs
        elif extension == 'rst':
            rst_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.RST, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            rst_docs = rst_splitter.create_documents([data])
            
            return rst_docs
        elif extension == 'rb':
            rb_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.RUBY, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            rb_docs = rb_splitter.create_documents([data])
            
            return rb_docs
        elif extension == 'rs':
            rs_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.RUST, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            rs_docs = rs_splitter.create_documents([data])
            
            return rs_docs
        elif extension == 'scala':
            scala_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.SCALA, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            scala_docs = scala_splitter.create_documents([data])
            
            return scala_docs
        elif extension == 'swift':
            swift_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.SWIFT, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            swift_docs = swift_splitter.create_documents([data])
            
            return swift_docs
        elif extension == 'md':
            # For Markdown, we use MarkdownHeaderTextSplitter because the CodeTextSplitter is not working well
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(data)
            return md_header_splits
        elif extension == 'tex':
            tex_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.LATEX, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            tex_docs = tex_splitter.create_documents([data])
            
            return tex_docs
        elif extension == 'html':
            # For HTML, we use HTMLHeaderTextSplitter
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
                ("h6", "Header 6"),
            ]

            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            html_header_splits = html_splitter.split_text(data)
            return html_header_splits
        
        elif extension == 'sol':
            sol_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.SOL, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            sol_docs = sol_splitter.create_documents([data])
            
            return sol_docs
        elif extension == 'cs':
            cs_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.CSHARP, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            cs_docs = cs_splitter.create_documents([data])
            
            return cs_docs
        elif extension == 'cob':
            cob_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.COBOL, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            cob_docs = cob_splitter.create_documents([data])
            
            return cob_docs
            
    elif extension in ['docx', 'xlsx', 'pptx', 'pdf', 'txt', 'csv']:
        # These extensions have not specific splitters so we use specific loader and recursive splitter
        
        # Load data
        documents = universal_loader(path)
        
        # Define recursive chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the documents with content aware chunking
        text = splitter.split_documents(documents)
        
    elif extension == 'json':
        # Load data
        documents = universal_loader(path)
            
        splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
        
        # Split the documents with content aware chunking
        text = splitter.create_documents([documents])
    else:
        print(f"Unsupported file extension: {extension}")
        return None
    
    return text

def semantic_chunking(path):
    """
        Function to do semantic chunking for one file
    """
    
    # Load data
    documents = universal_loader(path)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Define recursive chunking
    text_splitter = SemanticChunker(embeddings)
    
    # Split the documents with recursive chunking
    text = text_splitter.split_documents(documents)
    
    return text

def concatenate_page_contents(documents):
    """
    Concatenate the page_content of each document in the list into one string.

    Parameters:
    documents (list): A list of Document objects.

    Returns:
    str: A string containing all the page_content.
    """
    return "".join(doc.page_content for doc in documents)

def split_into_sentences(text):
    """
        Function to split one string to a list of sentences
    """

    sentences = re.findall(r'[^.!?]*[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences if sentences else [text]

def Agentic_chunking(path,openai_api_key):
    """
        Function to do agentic chunking for one file
    """

    # Load data
    document_data= universal_loader(path)
    data = concatenate_page_contents(document_data)
    
    # Split the data by sentences
    list_sentences_data = split_into_sentences(data)
    
    # Define agentic chuncker
    ac = AgenticChunker(openai_api_key)
    
    # Add sentences
    ac.add_propositions(list_sentences_data)
    
    # Print the chunck
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    
    # Return the result
    return ac.get_chunks(get_type='list_of_strings')


def get_file_paths_in_folder(folder_path):
    """
        Function to get the list of file path in a folder
    """

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def chunking_from_folder(folder_path, chunk_size=None, chunk_overlap=None, type_chunking="Fixed sized chunking", openai_api_key=None):
    """
    This function performs chunking on all files in a given folder using the specified chunking method.

    Parameters:
    - folder_path (str): The path to the folder containing the files to be chunked.
    - chunk_size (int, optional): The size of each chunk. Required for fixed size, recursive, and content-aware chunking methods.
    - chunk_overlap (int, optional): The number of overlapping elements between consecutive chunks. Required for fixed size, recursive, and content-aware chunking methods.
    - type_chunking (str, optional): The type of chunking method to use. Options are "Fixed sized chunking", "Recursive chunking", "Content-aware chunking", "Semantic chunking", and "Agentic chunking". Default is "Fixed sized chunking".
    - openai_api_key (str, optional): The API key for OpenAI. Required for the "Agentic chunking" method.

    Returns:
    - list_documents (list): A list of documents, where each document is a chunk of text from the files in the folder.

    Raises:
    - ValueError: If the specified chunking method is not one of the available options.
    """
    
    # Mapping the chunking functions to their names
    chunking_methods = {
        "Fixed sized chunking": fixed_size_chunking,
        "Recursive chunking": recursive_chunking,
        "Content-aware chunking": content_aware_chunking,
        "Semantic chunking": semantic_chunking,
        "Agentic chunking": Agentic_chunking
    }

    list_file_paths = get_file_paths_in_folder(folder_path)
    list_documents = []

    if type_chunking not in chunking_methods:
        print("Invalid choice. Please choose a valid chunking method.")
        return list_documents

    # Get the appropriate chunking function
    chunking_function = chunking_methods[type_chunking]

    for i, path in enumerate(list_file_paths):
        if type_chunking == "Semantic chunking":
            list_documents.extend(chunking_function(path))
        elif type_chunking == "Agentic chunking":
            list_documents.extend(chunking_function(path, openai_api_key))
        else:
            list_documents.extend(chunking_function(path, chunk_size, chunk_overlap))

    return list_documents

def create_db_from_document(name_db,texts):
    """
    This function creates a local database of document embeddings using the HuggingFace Embeddings and FAISS.

    Parameters:
    - name_db (str): The name of the database to be created.
    - texts (list): A list of documents

    Returns:
    None. The function saves the created database locally in the database user directory.

    Note:
    The function uses the 'sentence-transformers/all-MiniLM-L6-v2' model from HuggingFace for creating embeddings.
    """

    # Create an embedding for the document
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # create and save the local database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(name_db)


def create_db_from_text(name_db,texts):
    """
    This function creates a local database of document embeddings using the HuggingFace Embeddings and FAISS.

    Parameters:
    - name_db (str): The name of the database to be created.
    - texts (list): A list of string text

    Returns:
    None. The function saves the created database locally in the database of user directory.

    Note:
    The function uses the 'sentence-transformers/all-MiniLM-L6-v2' model from HuggingFace for creating embeddings.
    """


    # Create an embedding for the document
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # create and save the local database
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(name_db)