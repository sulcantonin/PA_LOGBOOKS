import sys,torch, os
from desy_wrapper import logbook_to_documents
from gradio_wrapper import *
from gradio import ChatInterface
import logging, configparser


logger = logging.getLogger("server")

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# ------- config -----------
config = configparser.ConfigParser()
config.read('../config/XFELelog.ini')
data_folder = config['sources']['data_folder']
data_file = config['sources']['data_file']
faiss_index = config['sources']['faiss_index']


# ------- preparations -----------
logger.info('loading embedding')
embedding = prepare_embedding()
logger.info('embedding loaded')

if os.path.isfile(f'{faiss_index}/index.faiss'):
    logger.info('loading vectorstore')
    vectorstore = FAISS.load_local(faiss_index, embedding, allow_dangerous_deserialization=True)
    logger.info('vectorstore loaded')
else:
    logger.info('loading dataset')
    data = logbook_to_documents(data_file)
    logger.info('dataset loaded')
    
    logger.info('preparing vectorstore')
    vectorstore = prepare_vectorstore(data.values(), embedding)
    logger.info('vectorstore prepared')
    
    logger.info('saving index')
    vectorstore.save_local(faiss_index)
    logger.info('index saved')


retriever = prepare_retriever(vectorstore)
logger.info('preparing llm')
tokenizer, model = prepare_llm(device = device)
logger.info('llm prepared')
# -------- gradio ------------
def logbook_chatbot(query, history):
    query, answer = generate(query, retriever, tokenizer, model, device)
    return answer

demo = ChatInterface(logbook_chatbot)
demo.launch(share = True)

