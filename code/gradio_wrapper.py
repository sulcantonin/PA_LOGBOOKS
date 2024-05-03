'''
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from hugginface_wrapper import HuggingFaceLLM
from time_retriever import FixedTimeWeightedVectorStoreRetriever
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import pickle, gzip
from datetime import datetime, timedelta
import sys
import outlines
from dateutil.parser import parse as date_parse
from tqdm.notebook import tqdm
gd = lambda x,i : x[list(x.keys())[i]]
'''
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from time_retriever import FixedTimeWeightedVectorStoreRetriever

import torch

def prepare_embedding(model_name = 'all-MiniLM-L6-v2'):
    # encode_kwargs = {'normalize_embeddings' : True}
    # distance_strategy = "MAX_INNER_PRODUCT"
    return HuggingFaceEmbeddings(model_name= model_name)# , encode_kwargs=encode_kwargs)
    

def prepare_vectorstore(documents : list[Document], embedding):
    vectorstore = FAISS.from_documents(documents, embedding)# , distance_strategy = distance_strategy)
    return vectorstore

def prepare_llm(model_id = 'mistralai/Mistral-7B-Instruct-v0.2', device = 'cuda'):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_id, 
                                                 device_map = device,
                                                 quantization_config = quantization_config)
    
    return tokenizer, model 
def prepare_retriever(vectorstore):
    '''
    retriever = FixedTimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, 
        decay_rate=0.5, 
        k=16, 
        timestamp_field = 'isodate',
        search_kwargs={"score_threshold": 0.5})
    codes = retriever.add_documents(data.values())
    '''
    return vectorstore.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={'score_threshold': 0.25, 'k' : 32})
    
def compile_query_for_tokenizer(query : str, retriever):
    # getting relevant documents 
    documents = retriever.invoke(query)
    
    # building a context for a LLM prompt
    context = "\n\n---\n".join([f"On {str(document.metadata['timestamp'])} happened: {document.page_content}" for document in documents])
    
    language = 'English'
    
    template =  f"""I want you to act as a question answering bot which uses the context mentioned and answer in
    a concise manner and doesn't make stuff up.
    You will answer question based on the context - {context}. 
    You will create content in {language} language.
    Question: {query}
    Answer:"""

    messages = [
        {"role": "user", "content": template}
    ]
    return messages, documents

def generate(query, retriever, tokenizer, model, device):
    
    messages, documents = compile_query_for_tokenizer(query, retriever)
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True)
    
    decoded = tokenizer.batch_decode(generated_ids)
    
    answer = decoded[0].split('[/INST]')[-1].split('</s')[0]
    
    return query, answer