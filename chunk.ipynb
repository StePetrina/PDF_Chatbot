{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PyPDF2 import PdfReader\n",
    "\n",
    "# Extracting Text from PDF\n",
    "def extract_text_from_pdf(file_path):\n",
    "    with open(file_path, 'rb') as file:\n",
    "        pdf = PdfReader(file)\n",
    "        text = \" \".join(page.extract_text() for page in pdf.pages)\n",
    "    return text\n",
    "\n",
    "# Extract text from the PDF and split it into sentences\n",
    "text = extract_text_from_pdf('DW.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ILIZZO QUOTIDIANO.................................................................................. 21\n",
      "11. CONSIGLI E SUGGERIMENTI UTILI...............................................................24\n",
      "12. MANUTENZIONE E PULIZIA........................................................................... 26\n",
      "13. RISOLUZIONE DEI PROBLEMI....................................................................... 30\n",
      "14. INFORMAZIONI TECNICHE.............................................................................36\n",
      "15. CONSIDERAZIONI SULL'AMBIENTE.............................................................. 37\n",
      "My AEG Kitchen app www.aeg.com 2 PER RISULTATI PERFETTI\n",
      "Grazie per aver scelto di acquistare questo prodotto AEG. Lo abbiamo creato per\n",
      "fornirvi prestazioni impeccabili per molti anni, grazie a tecnologie innovative che\n",
      "vi semplificheranno la vita - funzioni che non troverete sulle normali\n",
      "apparecchiature. Vi invitiamo di dedicare qualche minuto alla lettura per sapere\n",
      "come trarre il massimo dal vostro elettrodomestico.\n",
      "Visitate il nostro sito web per:\n",
      "Ricevere consigli, scaricare i nostri opuscoli, eliminare eventuali anomalie,\n",
      "ottenere informazioni sull'assistenza e la riparazione:\n",
      "www.aeg.com/support\n",
      "Per registrare il vostro prodotto e ricevere un servizio migliore:\n",
      "www.registeraeg.com\n",
      "Acquistare accessori, materiali di consumo e ricambi originali per la vostra\n",
      "apparecchiatura:\n",
      "www.aeg.com/shop\n",
      "ASSISTENZA CLIENTI E ASSISTENZA TECNICA\n",
      "Consigliamo sempre l’impiego di ricambi originali.\n",
      "Quando si contatta il nostro Centro di Assistenza Autorizzato, accertarsi di avere\n",
      "a disposizione i dati seguenti: Modello, numero dell’apparecchio (PNC), numero\n",
      "di serie.\n",
      "Le informazioni sono riportate sulla targhetta identificativa.\n",
      " Avvertenza/Attenzione - Importanti Informazioni per la sicurezza\n",
      " Informazioni generali e suggerimenti\n",
      " Informazioni ambientali\n",
      "Con riserva di modifiche.\n",
      "1. \n",
      " INFORMAZIONI DI SICUREZZA\n",
      "Leggere attentamente le istruzioni fornite prima di\n",
      "installare e utilizzare l'apparecchi\n"
     ]
    }
   ],
   "source": [
    "sample = text[1015:3037]\n",
    "print(sample)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### NLTK Sentence Tokenizer ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     /Users/stefanopetrina/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')\n",
    "\n",
    "# Splitting Text into Sentences\n",
    "def split_text_into_sentences(text):\n",
    "    sentences = nltk.sent_tokenize(text)\n",
    "    return sentences\n",
    "\n",
    "sentences = split_text_into_sentences(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The text has 70369 characters of length.\n",
      "Sentences extracted: 817\n",
      "Mean of 86.13096695226439 characters per sentence.\n"
     ]
    }
   ],
   "source": [
    "print('The text has', len(text), 'characters of length.')\n",
    "print('Sentences extracted:', len(sentences))\n",
    "print('Mean of', len(text)/len(sentences), 'characters per sentence.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Spacy Sentence Splitter ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "\n",
    "nlp = spacy.load('en_core_web_sm')\n",
    "doc = nlp(text)\n",
    "sentences = list(doc.sents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sentences extracted: 933\n",
      "Mean of 75.42229367631298 characters per sentence.\n"
     ]
    }
   ],
   "source": [
    "print('Sentences extracted:', len(sentences))\n",
    "print('Mean of', len(text)/len(sentences), 'characters per sentence.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Langchain Character Text Splitter ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "# Initialize the text splitter with custom parameters\n",
    "custom_text_splitter = RecursiveCharacterTextSplitter(\n",
    "    # Set custom chunk size\n",
    "    chunk_size = 100,\n",
    "    chunk_overlap  = 20,\n",
    "    # Use length of the text as the size measure\n",
    "    length_function = len\n",
    "\n",
    ")\n",
    "\n",
    "# Create the chunks\n",
    "sentences = custom_text_splitter.create_documents([sample])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "### Chunk 1: \n",
      "\n",
      "di serie.\n",
      "Le informazioni sono riportate sulla targhetta identificativa.\n",
      "\n",
      "=====\n",
      "\n",
      "### Chunk 2: \n",
      "\n",
      "Avvertenza/Attenzione - Importanti Informazioni per la sicurezza\n",
      "\n",
      "=====\n"
     ]
    }
   ],
   "source": [
    "# Print the first two chunks\n",
    "print(f'### Chunk 1: \\n\\n{sentences[25].page_content}\\n\\n=====\\n')\n",
    "print(f'### Chunk 2: \\n\\n{sentences[26].page_content}\\n\\n=====')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sentences extracted: 30\n",
      "Mean of 2345.633333333333 characters per sentence.\n"
     ]
    }
   ],
   "source": [
    "print('Sentences extracted:', len(sentences))\n",
    "print('Mean of', len(text)/len(sentences), 'characters per sentence.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the text splitter with custom parameters\n",
    "custom_text_splitter = RecursiveCharacterTextSplitter(\n",
    "    # Set custom chunk size\n",
    "    chunk_size = 300,\n",
    "    chunk_overlap  = 30,\n",
    "    # Use length of the text as the size measure\n",
    "    length_function = len,\n",
    "    # Use only \"\\n\\n\" as the separator\n",
    "    separators = ['\\n']\n",
    ")\n",
    "\n",
    "# Create the chunks\n",
    "custom_texts = custom_text_splitter.create_documents([sample])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sentences extracted: 8\n",
      "Mean of 8796.125 characters per sentence.\n"
     ]
    }
   ],
   "source": [
    "print('Sentences extracted:', len(custom_texts))\n",
    "print('Mean of', len(text)/len(custom_texts), 'characters per sentence.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sentences1 = [\"This is an example sentence.\", \"Another sentence goes here.\", \"...\"]\n",
    "# print(type(sentences1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### KMeans Clustering ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/homebrew/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1416: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  super()._check_params_vs_input(X, default_n_init=10)\n"
     ]
    }
   ],
   "source": [
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "# Load the Sentence Transformer model\n",
    "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
    "\n",
    "# Define a list of sentences (your text data)\n",
    "sentences = [\"This is an example sentence.\", \"Another sentence goes here.\", \"...\"]\n",
    "\n",
    "# Generate embeddings for the sentences\n",
    "embeddings = model.encode(sentences)\n",
    "\n",
    "# Choose an appropriate number of clusters (here we choose 5 as an example)\n",
    "num_clusters = 3\n",
    "\n",
    "# Perform K-means clustering\n",
    "kmeans = KMeans(n_clusters=num_clusters)\n",
    "clusters = kmeans.fit_predict(embeddings)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /Users/stefanopetrina/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "We need at least 1 word to plot a word cloud, got 0.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[44], line 33\u001b[0m\n\u001b[1;32m     30\u001b[0m cleaned_sentences \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(clean_sentence(s)) \u001b[38;5;28;01mfor\u001b[39;00m s \u001b[38;5;129;01min\u001b[39;00m cluster_sentences]\n\u001b[1;32m     31\u001b[0m text \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(cleaned_sentences)\n\u001b[0;32m---> 33\u001b[0m wordcloud \u001b[38;5;241m=\u001b[39m \u001b[43mWordCloud\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmax_font_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m50\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmax_words\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m100\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mbackground_color\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mwhite\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgenerate\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtext\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     34\u001b[0m plt\u001b[38;5;241m.\u001b[39mfigure()\n\u001b[1;32m     35\u001b[0m plt\u001b[38;5;241m.\u001b[39mimshow(wordcloud, interpolation\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbilinear\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[0;32m/opt/homebrew/lib/python3.11/site-packages/wordcloud/wordcloud.py:642\u001b[0m, in \u001b[0;36mWordCloud.generate\u001b[0;34m(self, text)\u001b[0m\n\u001b[1;32m    627\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mgenerate\u001b[39m(\u001b[38;5;28mself\u001b[39m, text):\n\u001b[1;32m    628\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Generate wordcloud from text.\u001b[39;00m\n\u001b[1;32m    629\u001b[0m \n\u001b[1;32m    630\u001b[0m \u001b[38;5;124;03m    The input \"text\" is expected to be a natural text. If you pass a sorted\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    640\u001b[0m \u001b[38;5;124;03m    self\u001b[39;00m\n\u001b[1;32m    641\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 642\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgenerate_from_text\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtext\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/opt/homebrew/lib/python3.11/site-packages/wordcloud/wordcloud.py:624\u001b[0m, in \u001b[0;36mWordCloud.generate_from_text\u001b[0;34m(self, text)\u001b[0m\n\u001b[1;32m    607\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Generate wordcloud from text.\u001b[39;00m\n\u001b[1;32m    608\u001b[0m \n\u001b[1;32m    609\u001b[0m \u001b[38;5;124;03mThe input \"text\" is expected to be a natural text. If you pass a sorted\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    621\u001b[0m \u001b[38;5;124;03mself\u001b[39;00m\n\u001b[1;32m    622\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    623\u001b[0m words \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprocess_text(text)\n\u001b[0;32m--> 624\u001b[0m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgenerate_from_frequencies\u001b[49m\u001b[43m(\u001b[49m\u001b[43mwords\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    625\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\n",
      "File \u001b[0;32m/opt/homebrew/lib/python3.11/site-packages/wordcloud/wordcloud.py:410\u001b[0m, in \u001b[0;36mWordCloud.generate_from_frequencies\u001b[0;34m(self, frequencies, max_font_size)\u001b[0m\n\u001b[1;32m    408\u001b[0m frequencies \u001b[38;5;241m=\u001b[39m \u001b[38;5;28msorted\u001b[39m(frequencies\u001b[38;5;241m.\u001b[39mitems(), key\u001b[38;5;241m=\u001b[39mitemgetter(\u001b[38;5;241m1\u001b[39m), reverse\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[1;32m    409\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(frequencies) \u001b[38;5;241m<\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m--> 410\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWe need at least 1 word to plot a word cloud, \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    411\u001b[0m                      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mgot \u001b[39m\u001b[38;5;132;01m%d\u001b[39;00m\u001b[38;5;124m.\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m \u001b[38;5;28mlen\u001b[39m(frequencies))\n\u001b[1;32m    412\u001b[0m frequencies \u001b[38;5;241m=\u001b[39m frequencies[:\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmax_words]\n\u001b[1;32m    414\u001b[0m \u001b[38;5;66;03m# largest entry will be 1\u001b[39;00m\n",
      "\u001b[0;31mValueError\u001b[0m: We need at least 1 word to plot a word cloud, got 0."
     ]
    }
   ],
   "source": [
    "from wordcloud import WordCloud\n",
    "import matplotlib.pyplot as plt\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "import string\n",
    "\n",
    "nltk.download('stopwords')\n",
    "\n",
    "# Define a list of stop words\n",
    "stop_words = set(stopwords.words('italian'))\n",
    "\n",
    "# Define a function to clean sentences\n",
    "def clean_sentence(sentence):\n",
    "    # Tokenize the sentence\n",
    "    tokens = word_tokenize(sentence)\n",
    "    # Convert to lower case\n",
    "    tokens = [w.lower() for w in tokens]\n",
    "    # Remove punctuation\n",
    "    table = str.maketrans('', '', string.punctuation)\n",
    "    stripped = [w.translate(table) for w in tokens]\n",
    "    # Remove non-alphabetic tokens\n",
    "    words = [word for word in stripped if word.isalpha()]\n",
    "    # Filter out stop words\n",
    "    words = [w for w in words if not w in stop_words]\n",
    "    return words\n",
    "\n",
    "# Compute and print Word Clouds for each cluster\n",
    "for i in range(num_clusters):\n",
    "    cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]\n",
    "    cleaned_sentences = [' '.join(clean_sentence(s)) for s in cluster_sentences]\n",
    "    text = ' '.join(cleaned_sentences)\n",
    "\n",
    "    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color=\"white\").generate(text)\n",
    "    plt.figure()\n",
    "    plt.imshow(wordcloud, interpolation=\"bilinear\")\n",
    "    plt.axis(\"off\")\n",
    "    plt.title(f\"Cluster {i}\")\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clustering Adjacent Sentences ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import spacy\n",
    "\n",
    "# Load the Spacy model\n",
    "nlp = spacy.load('en_core_web_sm')\n",
    "\n",
    "def process(text):\n",
    "    doc = nlp(text)\n",
    "    sents = list(doc.sents)\n",
    "    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])\n",
    "\n",
    "    return sents, vecs\n",
    "\n",
    "def cluster_text(sents, vecs, threshold):\n",
    "    clusters = [[0]]\n",
    "    for i in range(1, len(sents)):\n",
    "        if np.dot(vecs[i], vecs[i-1]) < threshold:\n",
    "            clusters.append([])\n",
    "        clusters[-1].append(i)\n",
    "    \n",
    "    return clusters\n",
    "\n",
    "def clean_text(text):\n",
    "    # Add your text cleaning process here\n",
    "    return text\n",
    "\n",
    "# Initialize the clusters lengths list and final texts list\n",
    "clusters_lens = []\n",
    "final_texts = []\n",
    "\n",
    "# Process the chunk\n",
    "threshold = 0.3\n",
    "sents, vecs = process(text)\n",
    "\n",
    "# Cluster the sentences\n",
    "clusters = cluster_text(sents, vecs, threshold)\n",
    "\n",
    "for cluster in clusters:\n",
    "    cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))\n",
    "    cluster_len = len(cluster_txt)\n",
    "    \n",
    "    # Check if the cluster is too short\n",
    "    if cluster_len < 60:\n",
    "        continue\n",
    "    \n",
    "    # Check if the cluster is too long\n",
    "    elif cluster_len > 3000:\n",
    "        threshold = 0.6\n",
    "        sents_div, vecs_div = process(cluster_txt)\n",
    "        reclusters = cluster_text(sents_div, vecs_div, threshold)\n",
    "        \n",
    "        for subcluster in reclusters:\n",
    "            div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))\n",
    "            div_len = len(div_txt)\n",
    "            \n",
    "            if div_len < 60 or div_len > 3000:\n",
    "                continue\n",
    "            \n",
    "            clusters_lens.append(div_len)\n",
    "            final_texts.append(div_txt)\n",
    "            \n",
    "    else:\n",
    "        clusters_lens.append(cluster_len)\n",
    "        final_texts.append(cluster_txt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "128\n",
      "1006\n",
      "1006\n"
     ]
    }
   ],
   "source": [
    "print(len(final_texts))\n",
    "print(len(final_texts[0]))\n",
    "print(clusters_lens[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1006, 2672, 2451, 64, 1371, 874, 335, 529, 502, 231, 1672, 2077, 1420, 530, 1499, 2176, 278, 398, 2431, 233, 189, 75, 91, 62, 241, 612, 730, 2301, 2505, 1336, 182, 130, 799, 162, 913, 122, 866, 88, 93, 75, 567, 83, 264, 234, 154, 66, 79, 156, 809, 333, 369, 222, 65, 641, 450, 539, 1880, 324, 716, 396, 1175, 885, 478, 217, 195, 81, 128, 917, 365, 167, 785, 157, 161, 120, 207, 331, 456, 64, 373, 106, 113, 122, 504, 155, 281, 181, 155, 817, 156, 247, 221, 75, 645, 166, 740, 264, 240, 960, 162, 74, 941, 62, 224, 62, 805, 92, 735, 1190, 128, 608, 771, 692, 312, 536, 490, 71, 132, 265, 1067, 126, 82, 127, 351, 206, 617, 185, 1790, 544]\n"
     ]
    }
   ],
   "source": [
    "final_texts_lengths = [len(chunk) for chunk in final_texts]\n",
    "print(final_texts_lengths)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
