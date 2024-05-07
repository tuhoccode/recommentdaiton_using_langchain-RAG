#code recommendation book
#langchain
#file csv
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


#1 import dataset
data_file = CSVLoader(file_path="book_genre.csv")
data_files = data_file.load()
#2 text_split
def cr_text():
    text_splitter = CharacterTextSplitter(
        chunk_size = 100, 
        chunk_overlap = 10,

    )
    texts = text_splitter.split_documents(data_files)
    #3 embedding
    embeddings = HuggingFaceEmbeddings(model_name = "google-bert/bert-base-uncased")
    db = FAISS.from_documents(texts, embeddings)
    return db
# global retriever
def cr_llm():
    key =  "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"
    llm = HuggingFaceHub(huggingfacehub_api_token=key, repo_id = "google/flan-t5-base", model_kwargs = {"temperature":0.1,"max_length":50})
    return llm
# global llm
def cr_promtpt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt
def RAG(prompt, llm,db):
    RAG = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type = 'stuff',
        retriever = db.as_retriever(),
        chain_type_kwargs = {"prompt": prompt}

        )
    return RAG
# global prompt
template = """Form a grammatically correct sentence as a response.
Compare the book given in question with others in the retriever based on genre
Return a complete sentence with the full title of the book and describe the sim
question: {question}
context: {context}"""
text = cr_text()
llm = cr_llm()
prompt = cr_promtpt(template)
QA = RAG(prompt, llm, text)

query = "Which books except 'The Lost Hero' are similar to it?"
out = QA.invoke(query)
print(out)

