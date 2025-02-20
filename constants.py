# Decorator
def constant(func):
    def func_set(self, value):
        raise TypeError

    def func_get(self):
        return func()
    return property(func_get, func_set)


# const class
class _Const(object):    
    @constant
    def DB_DIR():
        return "./db" 
    @constant
    def NM_DUCKDB():  
        return "clinical_study.duckdb"
    @constant
    def NM_BM25():  
        return "content_df.csv"


    @constant
    def MODEL_CHAT():  # 발화 모델
        return "gpt-4o-mini"
    @constant
    def OPENAI_KEY(): # OpenAI API Key 
        return ""
    

    @constant
    def HTML_SEARCH():
        return "./templates/search.html" 
    @constant
    def HTML_RESULTS():
        return "./templates/results.html"
    @constant
    def HTML_NODATA():
        return "./templates/nodata.html"
    
    @constant
    def PATH_EXAMPLE(): # llm shot example file path
        return "oneshot_example.json"


CONST = _Const()
