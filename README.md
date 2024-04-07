# Logbooks
This repo aims to facilitate the sharing of information, models, and datasets concerning logbooks in domain of particle accelerators. It serves as a hub for exchanging ideas, tools, and experiences 
by fostering an open and inclusive environment.

With the rapid advancements in NLP, this repository serves as a hub for exploring the potential applications of LLMs in logbook domains.  
By sharing datasets, model architectures, and evaluation methodologies, we aim to minimize duplicity. 

## Embedding
Two models available at HuggingFace
- PA_EMBEDDING_CASED Cased embedding, trained on PA_ARXIV, PA_JACOW and PA_BOOKS with no filtering of too long/short tokens, token window 266 tokens max, https://huggingface.co/sulcan/PA_EMBEDDING_CASED
- PA_EMBEDDING_UNCASED Uncased embedding, trained on PA_ARXIV, PA_JACOW and PA_BOOKS with filtering of too long (>=512)/short(<=16) tokens, token window 512 tokens max, https://huggingface.co/sulcan/PA_EMBEDDING_UNCASED


```python
from sentence_transformers import SentenceTransformer
sentences = ['ouch, I have a cavity in my tooth', 'superconducting cavity', 'cavity detuned']

model = SentenceTransformer('{MODEL_NAME}')
embeddings = model.encode(sentences)
print(embeddings)
```

- Tianyu Gao, Xingcheng Yao, Danqi Chen; SimCSE: Simple Contrastive Learning of Sentence Embeddings https://arxiv.org/abs/2104.08821

## RetriveQA

## Dataset
### ARXIV
[HuggingFace](https://huggingface.co/datasets/sulcan/PA_ARXIV)
### JACOW
[HuggingFace](https://huggingface.co/datasets/sulcan/PA_JACOW/tree/main)
### BOOKS
TBD
