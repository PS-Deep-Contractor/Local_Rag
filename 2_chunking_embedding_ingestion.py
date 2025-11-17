#################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import json
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import time


load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   DELETE CHROMA DB IF EXISTS AND INITIALIZE   ##################################################################################

import chromadb

# Delete old Chroma DB if exists
db_path = os.getenv("DATABASE_LOCATION")
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# NEW: create a PersistentClient
client = chromadb.PersistentClient(path=db_path)

# Initialize vector store with the new client
vector_store = Chroma(
    client=client,
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings
)


vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

###############################   INITIALIZE TEXT SPLITTER   ###################################################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

#################################################################################################################################################################
###############################   2.  PROCESSING THE JSON RESPONSE LINE BY LINE   ###############################################################################
#################################################################################################################################################################

###############################   FUNCTION TO EXTRACT RESPONSE LINE BY LINE   ###################################################################################



def process_json_lines(file_path):
    """Process each JSON line and extract relevant information."""
    extracted = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            extracted.append(obj)

    return extracted

file_content = process_json_lines(os.getenv("DATASET_STORAGE_FOLDER")+"data.txt")


#################################################################################################################################################################
###############################   3.  CHUNKING, EMBEDDING AND INGESTION   #######################################################################################
##################################################################################################################################################################

for line in file_content:
    try:
        # 1. Extract Metadata and Content based on the new JSON structure
        url = line['url']
        title = line['name'] # Using 'name' as the title

        # Combine the main description fields for comprehensive content
        full_description = line.get('full_description', '')
        about_text = line.get('about', '')
        
        # Create the raw text by combining available fields
        raw_text = f"Company: {title}. URL: {url}. About: {about_text}. Full Description: {full_description}"
        
        # 2. Validation Check
        if not raw_text or len(raw_text) < 50:
            print(f"Skipping {url}: Combined content too short or missing.")
            continue

        print(f"Processing URL: {url}")

        # 3. Create documents and split them
        texts = text_splitter.create_documents(
            [raw_text],
            metadatas=[{"source": url, "title": title}]
        )

        uuids = [str(uuid4()) for _ in range(len(texts))]

        # 4. Add to vector store
        vector_store.add_documents(documents=texts, ids=uuids)
        
    except KeyError as e:
        # Catches the error if 'url' or 'name' (used as title) is missing
        print(f"Skipping malformed document due to missing required key: {e}")
        continue