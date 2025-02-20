from flask import Flask, render_template_string, request, redirect, url_for
import duckdb
import pandas as pd
from constants import CONST
from search_trials import search_duckdb, search_bm25, get_trial_contents, get_chat_response, parse_trial_summary



app = Flask(__name__)

@app.route('/')
def home():
    return redirect(url_for('search'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Capture user input
        search_criteria = {
            "phase": request.form.get("phase"),
            "allocation": request.form.get("allocation"),
            "intervention_model": request.form.get("intervention_model"),
            "masking": request.form.get("masking"),
            "conditions": request.form.get("conditions"),
            "gender": request.form.get("gender"),
            "minimum_age": request.form.get("minimum_age"),
            "maximum_age": request.form.get("maximum_age"),
            "criteria": request.form.get("criteria"),
            "intervention": request.form.get("intervention"),
            "evaluation": request.form.get("evaluation"),
            "start_date": request.form.get("start_date"),
            "completion_date": request.form.get("completion_date"),
            "location_country": request.form.get("location_country"),
        }
        return redirect(url_for('results', **search_criteria))
    
    # Use render_template_string instead of a separate HTML file
    with open(CONST.HTML_SEARCH, 'r', encoding="utf-8") as file:
        html_template = file.read()

    # country list 생성 (frontend에서 사용)
    with duckdb.connect(CONST.PATH_DUCKDB) as conn:
        query = """
        SELECT location_countries FROM clinical_study GROUP BY location_countries
        """
        location_country = conn.execute(query).fetchdf()
    countries = set()
    for country in location_country["location_countries"]:
        if country:
            countries.update([country_nm.strip() for country_nm in country.split("; ")])
    countries = sorted(list(countries))

    return render_template_string(html_template, countries=countries)

@app.route('/results')
def results():
    search_criteria = request.args.to_dict()
    print("# search criteria")
    for key, value in search_criteria.items():
        print(f"{key}: {value}")
    trial_ids = search_duckdb(search_criteria)
    print("# sql searched num:", len(trial_ids))
    if len(trial_ids)==0: # 검색 결과가 없을 경우 no data 페이지로 이동
        with open(CONST.HTML_NODATA, 'r', encoding="utf-8") as file:
            html_template = file.read()
        return render_template_string(html_template)
    
    trial_ids = search_bm25(trial_ids, search_criteria)
    print("# bm25 searched trials:", trial_ids)
    contents_list = get_trial_contents(trial_ids)
    contents_text_list = []
    for i, res in enumerate(contents_list):
        content = f"## {i} nct_id: {res['nct_id']}\n"
        content += f"- **title**: {res['brief_title']}\n"
        content += f"- **summary**: {res['brief_summary']}\n"
        content += f"- **keywords**: {res['keywords']}\n"
        content += f"- **overall_status**: {res['overall_status']}\n"
        content += f"- **phase**: {res['phase']}\n" 
        content += f"- **study_type**: {res['study_type']}\n"
        content += f"- **allocation**: {res['allocation']}\n"
        content += f"- **intervention_model**: {res['intervention_model']}\n"
        content += f"- **masking**: {res['masking']}\n"
        content += f"- **study_design**: {res['study_design_text']}\n"
        content += f"- **conditions**: {res['conditions']}\n"
        content += f"- **criteria**: {res['criteria']}\n" 
        content += f"- **interventions**: {res['interventions_text']}"
        content += f"- **outcome_measures**: {res['outcomes']}\n"
        content += f"- **start_date**: {res['start_date']}\n"
        content += f"- **completion_date**: {res['completion_date']}\n"
        contents_text_list.append(content)
    response = get_chat_response(contents_text_list)

    trials = []
    for contents in contents_list:
        if contents['nct_id'] in response["trials"]:
            trials.append(response["trials"][contents['nct_id']])
            trials[-1]["nct_id"] = contents["nct_id"]
            trials[-1]["brief_title"] = contents["brief_title"]
            trials[-1]["url"] = contents["url"]
            trials [-1]["success"] = True if trials[-1]["성공/실패"] == "성공" else False

    with open(CONST.HTML_RESULTS, 'r', encoding="utf-8") as file:
        html_template = file.read()
    return render_template_string(html_template, trials=trials, proposal=response["proposal"])


if __name__ == '__main__':
    app.run(debug=True)