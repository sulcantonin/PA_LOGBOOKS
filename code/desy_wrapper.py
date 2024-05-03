import gzip, pickle
from langchain_core.documents import Document
from datetime import datetime, timedelta
from tqdm import tqdm

def logbook_to_documents(filename : str):
    with open(filename,'rb') as f:
        data_raw = pickle.load(f)

    print('loading dataset')
    
    data = {}
    for k in tqdm(data_raw):
        if 'text' in data_raw[k] and 'isodate' in data_raw[k] and 'time' in data_raw[k]:
            # logbook entry content
            text = data_raw[k]['text']
            if len(text) == 0:
                continue

            # logbook entry date
            isodate = datetime.strptime(data_raw[k]['isodate'],'%Y-%m-%d')

            # logbook entry time
            try:
                time = date_parse(data_raw[k]['time'], fuzzy = True)
                time = timedelta(hour = time.hour, minute = time.minute, second = time.second)
            except:
                time = timedelta(0)

            # date + time 
            timestamp = isodate + time

            if time.total_seconds() > 0.0:
                page_content = f'On {isodate.strftime("%d %B %Y")} at {str(time)} happened: {text}'
            else:
                page_content = f'On {isodate.strftime("%d %B %Y")} happened: {text}'

            # creating an entry
            data[k] = Document(
                page_content=page_content, 
                metadata=dict(isodate = isodate, 
                              timestamp = timestamp,
                              filename = '/'.join(k.split('/')[-2:])))
    
    print('dataset loaded')
    
    return data