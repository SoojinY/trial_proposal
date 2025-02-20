import requests
import zipfile
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import duckdb
from datetime import datetime
from constants import CONST 

DATA_URL = "https://clinicaltrials.gov/AllPublicXML.zip"
DATA_ZIP = "AllPublicXML.zip"
DATA_DIR = "data"


def download_clinicaltrials(url=DATA_URL, zip_file=DATA_ZIP):
    # 데이터 다운로드
    response = requests.get(url)
    # 파일 저장
    with open(zip_file, "wb") as file:
        file.write(response.content)
    print(f"Download complete! : {zip_file}")

def unzip_clinicaltrials(zip_file=DATA_ZIP, data_dir=DATA_DIR):
    # XML 파일 압축 풀기
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # 디렉터리가 없으면 생성
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"'{data_dir}' directory created!")
        zip_ref.extractall(path=data_dir)

def parse_xml(data_path)->dict:
    def convert_date_format(date_str):
        if not date_str:
            return None 
        try:
            # Parse "Month Year" format
            date_obj = datetime.strptime(date_str, "%B %Y")
            # Convert to "yyyy.MM" format
            return date_obj.strftime("%Y.%m")
        except ValueError:
            return None  

    xml_data = {}
    # XML 파일 파싱
    tree = ET.parse(data_path)
    root = tree.getroot()
    # XML 데이터 추출
    xml_data["nct_id"] = root.findtext("id_info/nct_id")
    xml_data["url"] = root.findtext("required_header/url")
    xml_data["brief_title"] = root.findtext("brief_title")
    xml_data["brief_summary"] = root.findtext("brief_summary/textblock").replace('\r\n', ' ') if root.find("brief_summary") is not None else None
    xml_data["phase"] = root.findtext("phase")
    xml_data["study_type"] = root.findtext("study_type")
    # Study design
    xml_data["allocation"] = root.findtext("study_design_info/allocation")
    xml_data["intervention_model"] = root.findtext("study_design_info/intervention_model")
    xml_data["masking"] = root.findtext("study_design_info/masking")
    # Conditions
    xml_data["condition_list"] = [condition.text for condition in root.findall("condition") if condition.text]
    # Enrollment
    xml_data["enrollment"] = root.findtext("enrollment")
    # Eligibility
    xml_data["gender"] = root.findtext("eligibility/gender")
    xml_data["minimum_age"] = root.findtext("eligibility/minimum_age")
    xml_data["maximum_age"] = root.findtext("eligibility/maximum_age")
    xml_data["criteria"] = root.findtext("eligibility/criteria/textblock").replace('\r\n', ' ') if root.find("eligibility/criteria") is not None else None
    # Interventions
    xml_data["interventions"] = [{"intervention_type": intervention.findtext("intervention_type"),
                                  "intervention_name": intervention.findtext("intervention_name")}
                                 for intervention in root.findall("intervention")]
    # Location Countries
    xml_data["location_countries"] = [country.text for country in root.findall("location_countries/country")]
    # Dates & Status
    xml_data["start_date"] = convert_date_format(root.findtext("start_date"))
    xml_data["completion_date"] = convert_date_format(root.findtext("primary_completion_date"))
    xml_data["primary_completion_date"] = convert_date_format(root.findtext("primary_completion_date"))
    xml_data["overall_status"] = root.findtext("overall_status")
    # Keywords
    xml_data["keywords"] = [keyword.text for keyword in root.findall("keyword")]
    
    outcomes_text = ""
    primary_outcome = root.findall("primary_outcome")
    if primary_outcome:
        outcomes_text += "[Primary Outcome]\n"
        for i, outcome in enumerate(primary_outcome):
            outcomes_text += "{}. {}\n".format(i+1, outcome.findtext("measure"))
            if outcome.findtext("time_frame"):
                outcomes_text += "- Time Frame: {}\n".format(outcome.findtext("time_frame"))
            if outcome.findtext("description"):
                outcomes_text += "- Description: {}\n".format(outcome.findtext("description"))
    secondary_outcome = root.findall("secondary_outcome")
    if secondary_outcome:
        outcomes_text += "[Secondary Outcome]\n"
        for i, outcome in enumerate(secondary_outcome):
            outcomes_text += "{}. {}\n".format(i+1, outcome.findtext("measure"))
            if outcome.findtext("time_frame"):
                outcomes_text += "- Time Frame: {}\n".format(outcome.findtext("time_frame"))
            if outcome.findtext("description"):
                outcomes_text += "- Description: {}\n".format(outcome.findtext("description"))
    other_outcome = root.findall("other_outcome")
    if other_outcome:
        outcomes_text += "[Other Outcome]\n"
        for i, outcome in enumerate(other_outcome):
            outcomes_text = "{}. {}\n".format(i+1, outcome.findtext("measure"))
            if outcome.findtext("time_frame"):
                outcomes_text += "- Time Frame: {}\n".format(outcome.findtext("time_frame"))
            if outcome.findtext("description"):
                outcomes_text += "- Description: {}\n".format(outcome.findtext("description"))
    xml_data["outcomes"] = outcomes_text
    
    return xml_data

def insert_xml(xml_data, conn, pd_data):
    nct_id = xml_data["nct_id"]
    url = xml_data["url"]
    brief_title = xml_data["brief_title"]
    brief_summary = xml_data["brief_summary"]
    phase = xml_data["phase"]
    study_type = xml_data["study_type"]
    allocation = xml_data["allocation"]
    intervention_model = xml_data["intervention_model"]
    masking = xml_data["masking"]
    study_design_text = "- allocation: {}; \n- intervention_model: {}; \n- masking: {};\n".format(allocation, intervention_model, masking)
    interventions_text = "- interventions: "+"; ".join([intervention["intervention_type"]+":"+intervention["intervention_name"] for intervention in xml_data["interventions"]])
    conditions = "; ".join(xml_data["condition_list"]) if xml_data["condition_list"] else None
    enrollment = xml_data["enrollment"]
    gender = xml_data["gender"]
    minimum_age = xml_data["minimum_age"]
    maximum_age = xml_data["maximum_age"]
    criteria = xml_data["criteria"]
    location_countries = "; ".join(xml_data["location_countries"]) if xml_data["location_countries"] else None
    start_date = xml_data["start_date"]
    completion_date = xml_data["completion_date"]
    primary_completion_date = xml_data["primary_completion_date"]
    overall_status = xml_data["overall_status"]
    keywords = "; ".join(xml_data["keywords"])
    outcomes = xml_data["outcomes"]

    # Insert into clinical_study table
    conn.execute("""
    INSERT INTO clinical_study VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(nct_id) DO UPDATE SET 
        url=excluded.url, 
        brief_title=excluded.brief_title, 
        brief_summary=excluded.brief_summary,
        phase=excluded.phase, 
        study_type=excluded.study_type, 
        allocation=excluded.allocation,
        study_design_text=excluded.study_design_text,
        intervention_model=excluded.intervention_model, 
        interventions_text=excluded.interventions_text,
        masking=excluded.masking, 
        conditions=excluded.conditions, 
        enrollment=excluded.enrollment, 
        gender=excluded.gender,
        minimum_age=excluded.minimum_age, 
        maximum_age=excluded.maximum_age, 
        criteria=excluded.criteria,
        location_countries=excluded.location_countries,
        start_date=excluded.start_date,
        completion_date=excluded.completion_date, 
        primary_completion_date=excluded.primary_completion_date,
        overall_status=excluded.overall_status,
        keywords=excluded.keywords,
        outcomes=excluded.outcomes
    """, (nct_id, url, brief_title, brief_summary, phase, 
        study_type, allocation, study_design_text, intervention_model, interventions_text, masking, 
        conditions, enrollment, gender, minimum_age, maximum_age, 
        criteria, location_countries, start_date, completion_date, primary_completion_date, 
        overall_status, keywords, outcomes))
    
    content = "[Interventions Text]\n{}\n[Conditions]\n{}\n[Criteria]\n{}[Outcomes]\n{}".format(
        interventions_text,
        conditions,
        criteria,
        outcomes)

    pd_data.append({
        'nct_id': nct_id,
        'content': content
    })

def insert_xml2faiss(pd_data, db_path, embedding_model, batch_size):
    # faiss rag 용 
    # Actually not used in this project but can be used for RAG
    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    
    # 임베딩 모델 로드
    embedding_function =  HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},  # cuda, cpu
        encode_kwargs={"normalize_embeddings": True},
    )
    # 임베딩 차원 크기를 계산
    dimension_size = len(embedding_function.embed_query("hello world"))
    print("dimension_size:", dimension_size)
    faiss_db = FAISS(
    embedding_function=embedding_function,
    index=faiss.IndexFlatL2(dimension_size),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    )
    nct_ids = pd_data["nct_id"].tolist()
    content = pd_data["content"].tolist()
    # Process in batches
    for i in tqdm(range(0, len(pd_data), batch_size), desc="Processing Batches"):
        batch_nct_ids = nct_ids[i : i + batch_size]
        batch_content = content[i : i + batch_size]
        # Convert to metadata format
        batch_metadata = [{"nct_id": nct_id} for nct_id in batch_nct_ids]
        batch_ids = [ int(nct_id.replace("NCT", "")) for nct_id in batch_nct_ids ]
        # Insert into faiss db
        faiss_db.add_texts(metadatas=batch_metadata, texts=batch_content, ids=batch_ids)
    print("Number of documents:", len(faiss_db.docstore._dict))
    faiss_db.save_local(db_path)
    print("Data saved to FAISS!:", db_path)

def generate_db(download_data=True, unzip_data=True, url=DATA_URL, zip_file=DATA_ZIP,
                data_dir=DATA_DIR, db_dir=CONST.DB_DIR, duckdb_name=CONST.NM_DUCKDB, bm25_name=CONST.NM_BM25,
                faiss_rag=False, faiss_db_path="faiss_index", faiss_embedding_model="intfloat/multilingual-e5-small", faiss_batch_size=1000):
    # 디렉터리가 없으면 생성
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"'{db_dir}' directory created!")
    if download_data:
        download_clinicaltrials(url, zip_file)
    if unzip_data:
        unzip_clinicaltrials(zip_file, data_dir)
    
    duckdb_path = os.path.join(db_dir, duckdb_name)
    bm25_path = os.path.join(db_dir, bm25_name)
    # DuckDB 연결
    conn = duckdb.connect(duckdb_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS clinical_study (
        nct_id VARCHAR PRIMARY KEY,
        url VARCHAR NULL,
        brief_title VARCHAR NULL,
        brief_summary TEXT NULL,
        phase VARCHAR NULL,
        study_type VARCHAR NULL,
        allocation VARCHAR NULL,
        study_design_text TEXT NULL,
        intervention_model VARCHAR NULL,
        interventions_text TEXT NULL,
        masking VARCHAR NULL,
        conditions VARCHAR NULL,
        enrollment VARCHAR NULL,
        gender VARCHAR NULL,
        minimum_age VARCHAR NULL,
        maximum_age VARCHAR NULL,
        criteria TEXT NULL,
        location_countries VARCHAR NULL,
        start_date VARCHAR NULL,
        completion_date VARCHAR NULL,
        primary_completion_date VARCHAR NULL,
        overall_status VARCHAR NULL,
        keywords VARCHAR NULL,
        outcomes TEXT NULL
    );""")
    pd_data = []
    # XML 파일 파싱
    for dir_name in tqdm(os.listdir(data_dir)):
        if os.path.isfile(os.path.join(data_dir, dir_name)): continue
        for item in os.listdir(os.path.join(data_dir, dir_name)):
            item_path = os.path.join(data_dir, dir_name, item)
            if os.path.isfile(item_path) and item.endswith(".xml"):
                xml_data=parse_xml(item_path)
                insert_xml(xml_data, conn, pd_data)
    conn.close()
    print("Data inserted into DuckDB!:", duckdb_path)
    df = pd.DataFrame(pd_data)
    df.to_csv(bm25_path, index=False)
    print("Data saved to CSV!:", bm25_path)
    if faiss_rag:
        print("Generating FAISS DB for RAG...")
        insert_xml2faiss(df, faiss_db_path, faiss_embedding_model, faiss_batch_size)


if __name__ == "__main__":
    generate_db()