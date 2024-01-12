### PDF Chatbot using MistralAI (from HuggingFace)
from dotenv import load_dotenv
load_dotenv()

### Load the PDF
from PyPDF2 import PdfReader
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

# Extract text from the PDF and split it into sentences
text = extract_text_from_pdf('DW.pdf')

### Split (Clustering Adjacent Sentences)
import numpy as np
import spacy

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')

def process(text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i-1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text

# Initialize the clusters lengths list and final texts list
clusters_lens = []
final_texts = []

# Process the chunk
threshold = 0.3
sents, vecs = process(text)

# Cluster the sentences
clusters = cluster_text(sents, vecs, threshold)

for cluster in clusters:
    cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
    cluster_len = len(cluster_txt)
    
    # Check if the cluster is too short
    if cluster_len < 60:
        continue
    
    # Check if the cluster is too long
    elif cluster_len > 3000:
        threshold = 0.6
        sents_div, vecs_div = process(cluster_txt)
        reclusters = cluster_text(sents_div, vecs_div, threshold)
        
        for subcluster in reclusters:
            div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
            div_len = len(div_txt)
            
            if div_len < 60 or div_len > 3000:
                continue
            
            clusters_lens.append(div_len)
            final_texts.append(div_txt)
            
    else:
        clusters_lens.append(cluster_len)
        final_texts.append(cluster_txt)

# List of chunks
sentences = final_texts

### Store (using FAISS)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

embeddings=HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = FAISS.from_texts(sentences, embeddings)

### LLM
import os
from langchain.llms import HuggingFaceHub
repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                    repo_id=repo_id,
                    model_kwargs={'temperature':0.75, 'max_length':00}
                    )

### Retrieve and Generate
question = "Cosa fare se il display mostra iC4 o iC5?"

# Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.get_relevant_documents(
    question
)

# Generate
from langchain.schema import StrOutputParser
from langchain.output_parsers.structured import StructuredOutputParser

# from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain.prompts import PromptTemplate
template = """You are an helpful Teaching Assistant.
    Use only the following pieces of context to answer the question at the end. 
    Avoid personal opinions or unverified advice. 
    Request clarification as needed to ensure relevant responses. 
    Communicate clearly and professionally. 
    In cases where you lack sufficient information in the provided context, 
    respond with: 'Sorry, I do not have enough information about it.'.
    Answer in italian.

    The context is: {context}

    Question: {question}

    Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
    # | StructuredOutputParser()
)
rag_chain.invoke(question)
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)