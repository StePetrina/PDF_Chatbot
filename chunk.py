from PyPDF2 import PdfReader

# Extracting Text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

# Extract text from the PDF and split it into sentences
text = extract_text_from_pdf('file.pdf')

### NLTK Sentence Tokenizer
import nltk
nltk.download('punkt')

# Splitting Text into Sentences
def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

sentences = split_text_into_sentences(text)

### Spacy Sentence Splitter
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
sentences = list(doc.sents)

### Langchain Character Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Initialize the text splitter with custom parameters
custom_text_splitter = RecursiveCharacterTextSplitter(
    # Set custom chunk size
    chunk_size = 100,
    chunk_overlap  = 20,
    # Use length of the text as the size measure
    length_function = len

)

# Create the chunks
sentences = custom_text_splitter.create_documents(text)

### Kmeans Clustering Adjacent Sentences
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
