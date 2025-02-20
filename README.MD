# Requirements
- python=3.10
- flask
- requests
- pandas
- rank_bm25
- langchain_community
- langchain-openai
- langchain_huggingface (for FAISS RAG)
- faiss-gpu or faiss-cpu (for FAISS RAG)


Fill in your OpenAI API Key in `constants.py > OPENAI_KEY()`:
```python
    @constant
    def MODEL_CHAT():  # 발화 모델
        return "gpt-4o-mini"
    @constant
    def OPENAI_KEY(): # OpenAI API Key 
        return "your OpenAI API Key"  
```


This program supports FAISS RAG but primarily uses BM25 for document retrieval. 
To use FAISS, you need to modify the default setting in the code.
For FAISS RAG, GPU environment is recommanded.

You can install these packages with command:
```bash
pip install -r requirements.txt
```

# How to Run
```bash
python app.py
```


If the dataset is not ready, run `preprocess.py` with command : `python preproccess.py`


### What `preprocess.py` Does?
1. Download `AllPublicXML.zip` from https://clinicaltrials.gov/AllPublicXML.zip
2. Unzip into `data` directory
3. Create DB (duckdb) and CSV file (for BM25 search) then insert parsed `.xml` files 