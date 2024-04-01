from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA


class Rag:
    """
    This class represents a RAG model for question answering.
    """

    def __init__(self, model, model_type='llama', device='cpu', length=256, creativity=0.1,batch_size=8,hf_key=None):
        """
        Initializes the Rag class with the given parameters.

        Args:
            model (str): The name of the model to be used.
            model_type (str, optional): The type of the model. Defaults to 'llama'.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            length (int, optional): The maximum number of new tokens for the model. Defaults to 256.
            creativity (float, optional): The temperature for the model. Defaults to 0.1.
        """
        config = {'max_new_tokens': length, 'temperature': creativity,'batch_size':batch_size}
        self.llm = CTransformers(model=model,
                                 model_type=model_type,
                                 config=config)
        
        if device.lower()=="gpu":
            device='cuda'

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device})
        
        
    def add_local_db(self,local_db):
        self.db = FAISS.load_local(local_db, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.db.as_retriever(search_kwargs={'k': 2})
        self.prompt = PromptTemplate(
            template="""Use the following pieces of information to answer the user's question.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Context: {context}
                        Question: {question}
                     """,
            input_variables=['context', 'question'])
        self.qa_llm = RetrievalQA.from_chain_type(llm=self.llm,
                                                  chain_type='stuff',
                                                  retriever=self.retriever,
                                                  return_source_documents=True,
                                                  chain_type_kwargs={'prompt': self.prompt})

    def ask(self, query, tone=None):
        """
        Asks a question using the RAG model.

        Args:
            query (str): The question to ask.
            tone (str, optional): The tone to use when asking the question.

        Returns:
            dict: The output object from the question.
        """
        if tone is None:
            prompt = f'"{query}"'
        else:
            prompt = f'"{query}" Use the {tone} tone to answer'
        output = self.qa_llm({'query': prompt})
        return output


    def augment(self,model_available,creativity_level,len_out):
        config = {'max_new_tokens': len_out, 'temperature': creativity_level,'batch_size':8}
        self.llm = CTransformers(model=model_available,
                                 model_type='llama',
                                 config=config)  
        self.qa_llm = RetrievalQA.from_chain_type(llm=self.llm,
                                                  chain_type='stuff',
                                                  retriever=self.retriever,
                                                  return_source_documents=True,
                                                  chain_type_kwargs={'prompt': self.prompt})