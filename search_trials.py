import duckdb
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from constants import CONST
import json

llm = ChatOpenAI(
    model=CONST.MODEL_CHAT,
    temperature=0,
    openai_api_key=CONST.OPENAI_KEY,
)
llm = llm.bind(response_format={"type": "json_object"})
content_df = pd.read_csv(CONST.PATH_BM25)

def search_duckdb(criteria: dict)->list:
    # Search the database for studies based on the search criteria
    # input: criteria (dict) - search criteria
    # output: list of nct_ids
    query = """
    SELECT nct_id FROM clinical_study WHERE 1=1
    """
    params = []
    
    if criteria.get("phase"):
        if criteria["phase"] == "None":
            query += " AND (phase IS NULL OR phase = 'N/A')" 
        else:
            query += " AND phase LIKE ?"
            params.append(f"%{criteria['phase']}%")

    if criteria.get("allocation"):
        if criteria["allocation"] == "None":
            query += " AND (allocation IS NULL OR allocation = 'N/A')"
        else:
            query += " AND allocation = ?"
            params.append(criteria["allocation"])

    if criteria.get("intervention_model"):
        if criteria["intervention_model"] == "None": 
            query += " AND (intervention_model IS NULL OR intervention_model = 'N/A')"
        else:
            query += " AND intervention_model = ?"
            params.append(criteria["intervention_model"])

    if criteria.get("masking"):
        if criteria["masking"] == "None":
            query += " AND (masking IS NULL OR masking = 'N/A')"
        else:
            query += " AND masking = ?"
            params.append(criteria["masking"])
    
    # ########text search
    # if criteria.get("conditions"):
    #     query += " AND conditions LIKE ?"
    #     params.append(f"%{criteria['conditions']}%")

    if criteria.get("gender"):
        if criteria["gender"] == "None":
            query += " AND gender IS NULL or gender = 'N/A'"
        else:
            query += " AND gender = ?"
            params.append(criteria["gender"])
    
    if criteria.get("minimum_age"):
        query += " AND (minimum_age >= ? AND minimum_age != 'N/A')"
        params.append("{} Years".format(criteria["minimum_age"]))
    
    if criteria.get("maximum_age"):
        query += " AND (maximum_age <= ? AND maximum_age != 'N/A')"
        params.append("{} Years".format(criteria["maximum_age"]))

    # ########text search
    # if criteria.get("criteria"):
    #     query += " AND criteria LIKE ?"
    #     params.append(f"%{criteria['criteria']}%")
    
    # ########text search
    # if criteria.get("intervention"):
    #     query += " AND interventions LIKE ?"
    #     params.append(f"%{criteria['intervention']}%")

    ########text search
    # if criteria.get("evaluation"):
    #     query += " AND evaluation LIKE ?"
    #     params.append(f"%{criteria['evaluation']}%")
    
    if criteria.get("start_date"):
        query += " AND start_date >= ?"
        params.append(criteria["start_date"])
    
    if criteria.get("completion_date"):
        query += " AND completion_date <= ?"
        params.append(criteria["completion_date"])
    
    if criteria.get("location_country"):
        query += " AND location_countries ILIKE ?"
        params.append(f"%{criteria['location_country']}%")
    
    with duckdb.connect(CONST.PATH_DUCKDB) as conn:
        results = conn.execute(query, params).fetchdf()
        conn.close()
    return results["nct_id"].tolist()


def search_bm25(nct_ids: list, criteria: dict)->list:
    # Search the BM25 index for relevant documents
    # input: nct_ids (list) - list of nct_ids
    #        criteria (dict) - search criteria
    # output: list of nct_ids
    filtered_df = content_df[content_df["nct_id"].isin(nct_ids)]
    documents = [
        Document(page_content=row['content'], metadata={"nct_id": row['nct_id']}) for _, row in filtered_df.iterrows()
    ]
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents, k=5)
    bm25_query =""
    if criteria.get("intervention"):
        bm25_query += "[Intervention Text]\n{}\n".format(criteria.get("intervention"))
    if criteria.get("conditions"):
        bm25_query += "[Conditions Text]\n{}\n".format(criteria.get("conditions"))
    if criteria.get("criteria"):
        bm25_query += "[Criteria]\n{}\n".format(criteria.get("criteria"))
    if criteria.get("evaluation"):
        bm25_query += "[Outcomes]\n{}\n".format(criteria.get("evaluation"))
    bm25_results = bm25_retriever.invoke(bm25_query)
    return [doc.metadata["nct_id"] for doc in bm25_results]

def search_faiss(nct_ids: list, criteria: dict, faiss_index, embedding_model)->list:
    # Search the FAISS index for relevant documents
    # input: nct_ids (list) - list of nct_ids
    #        criteria (dict) - search criteria
    # output: list of nct_ids

    from langchain_huggingface import HuggingFaceEmbeddings
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore

    embedding_function = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},  # cuda, cpu
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(
        folder_path=faiss_index,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True,
    )
    faiss_query =""
    if criteria.get("intervention"):
        faiss_query += "[Intervention Text]\n{}\n".format(criteria.get("intervention"))
    if criteria.get("conditions"):
        faiss_query += "[Conditions Text]\n{}\n".format(criteria.get("conditions"))
    if criteria.get("criteria"):
        faiss_query += "[Criteria]\n{}\n".format(criteria.get("criteria"))
    if criteria.get("evaluation"):
        faiss_query += "[Outcomes]\n{}\n".format(criteria.get("evaluation"))
    results = db.similarity_search(faiss_query, k=5,filter = {"nct_id": {"$in":nct_ids}})
    return [doc.metadata["nct_id"] for doc in results]

    
def get_trial_contents(nct_ids: list)->list:
    # Fetch the clinical trials based on the nct_ids
    # input: nct_ids (list) - list of nct_ids
    # output: list of trial contents
    query = """
    SELECT * FROM clinical_study WHERE nct_id IN ({})
    """.format(", ".join(["?"]*len(nct_ids)))
    with duckdb.connect(CONST.PATH_DUCKDB) as conn:
        results = conn.execute(query, nct_ids).fetchdf().to_dict(orient="records")
        conn.close()
    return results
    

def get_chat_response(contents_list)->dict:
    # Generate a chat response based on the trial contents
    # input: contents_list (list) - list of trial contents
    # output: chat response
    with open(CONST.PATH_EXAMPLE, 'r', encoding="utf-8") as file:
        example = file.read()
    examples = [{"output": example}]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human", 
                "주어진 문서를 각각 요약하고 분석해서 새로운 임상시험을 제안하세요."
                "요약 시, 임상시험의 성공/실패 여부를 추론하고 주요 결과를 요약하세요." 
                "성공/실패 여부는 '성공' 또는 '실패'로만 표기합니다. "
                "proposal은 ALLOCATION, INTERVENTION_MODEL, MASKING, INTERVENTIONS, 대조군 구성, ENROLLMENT 등 임상시험 핵심 요건을 간단히 요약하여 제공합니다. "
                "답변은 json 형식으로 입력하세요."
            ),
            (
                "ai", "{output}"
            ) 
        ]
    )
    fewshot_prompt = FewShotChatMessagePromptTemplate(examples=examples,example_prompt=example_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 도움이 되는 임상시험 분석가입니다. "
                "제공된 임상시험 문서들의 성공/실패 여부를 추론하고 주요 결과를 요약하세요. "
                "임상시험들을 분석해서 새로운 임상시험을 제안하고 성공 확률과 주요 리스크를 설명하세요. "
                "답변은 json 형식으로 입력하세요."
            ),
            fewshot_prompt,
            (
                "human", 
                "주어진 문서를 각각 요약하고 분석해서 새로운 임상시험을 제안하세요."
                "\n문서: {contents}"
                "요약 시, 임상시험의 성공/실패 여부를 추론하고 주요 결과를 요약하세요." 
                "성공/실패 여부는 '성공' 또는 '실패'로만 표기합니다. "
                "제안은 ALLOCATION, INTERVENTION_MODEL, MASKING, INTERVENTIONS, 대조군 구성, ENROLLMENT 등 임상시험 핵심 요건을 간단히 요약하여 제공합니다. "
                "답변은 json 형식으로 입력하세요."
            )
        ] 
    )
    chain = prompt | llm | StrOutputParser()
    contents = "---------------------------\n".join(contents_list)
    chat_response = chain.invoke({"contents": contents})
    # Save the chat response to a file (확인용)
    with open('output.txt', 'w', encoding="utf-8") as file:
        file.write(prompt.format(contents=contents))
        file.write("\n\n++++++++++++++++++++++++++\n\n")
        file.write(chat_response)
    return json.loads(chat_response)

def parse_trial_summary(text:str)->list:
    # Parse the trial summary to extract key information
    # input: text (str) - trial summary
    # output: list of key information
    header_cnt = text.split("\n")[-1].count("#")
    trial_summary = []
    for trial_section in text.strip().split("#"*(header_cnt+1)+" ")[1:]:
        lines = trial_section.strip().split("\n")
        nct_id = lines[0].split(". ")[1].strip()
        lines = "\n".join(lines[1:])
        for line in lines.split("- **"):
            if "성공/실패" in line:
                success = "성공" in line.split("**: ",1)[1].strip()
            elif "결과 요약" in line:
                summary = line.split("**: ",1)[1]
                if "#"*header_cnt+" " in summary:
                    summary = summary.split("#"*header_cnt+" ",1)[0].strip()
        print(nct_id, success)
        print(summary)
        trial_summary.append({nct_id: {"success": success, "summary": summary}})
    return trial_summary