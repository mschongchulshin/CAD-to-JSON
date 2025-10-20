# app.py

import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd

st.set_page_config(
    page_title="AI Drawing Analysis Demo",
    layout="wide"
)

st.title("AI 기반 CAD 도면 JSON 변환")
st.markdown("---")

@st.cache_data
def load_training_data(filepath):
    try:
        # 파일이 UTF-8로 저장되었으므로, 간단하게 utf-8로만 읽습니다.
        df = pd.read_csv(filepath, encoding='utf-8')
        train_jsons = []
        for index, row in df.iterrows():
            json_string = row['JSON']
            try:
                # CSV 파일 저장 시 따옴표가 중복되는 문제를 해결합니다.
                corrected_json_string = json_string.replace('""', '"')
                train_jsons.append(json.loads(corrected_json_string))
            except json.JSONDecodeError as e:
                # 오류가 있는 데이터는 건너뛰고 터미널에 경고를 출력합니다.
                print(f"경고: CSV 파일의 {index + 2}번째 행의 JSON 데이터를 파싱하는 데 실패했습니다. 오류: {e}")
                continue
        return train_jsons
    except Exception as e:
        st.error(f"학습 데이터를 로딩하는 중 오류가 발생했습니다: {e}")
        return None

@st.cache_data
def create_few_shot_prompt(train_jsons):
    if not train_jsons:
        st.error("프롬프트를 생성할 학습 데이터가 없습니다.")
        return ""

    prompt = """You are an expert in analyzing factory layout drawings to generate accurate JSON.

# Information to Extract (Positional data is critical!)
1. **meta**: grid_unit (10x10), origin (R1.position)
2. **robots**: id, position [x,y], max_reach, work_angles [start_angle, end_angle]
3. **conveyors**: id, center [x,y], size [width, height], direction (X/Y/-X/-Y), shape
4. **tables**: id, center [x,y], size [width, height]
5. **equipments**: id, type, center [x,y], size, orientation_deg, attached_to
6. **fences/safety_fence**: coordinate info (start [x,y], end [x,y] or center, size)
7. **tasks/work_sequences**: work order

# Coordinate System (Very Important!)
- The origin (0,0) is the position of R1.
- The unit is a 10x10 grid (1 square = 10 actual units).
- X-axis: Positive to the right, negative to the left.
- Y-axis: Positive upwards, negative downwards.
- All coordinates are relative to R1.
"""
    prompt += f"\n# Reference Examples ({len(train_jsons)})\n\n"
    for i, json_data in enumerate(train_jsons):
        prompt += f"Example {i+1}:\n```json\n{json.dumps(json_data, ensure_ascii=False, indent=2)}\n```\n\n"
    prompt += """
# Final Instructions
1. Accurately identify all elements in the drawing.
2. Measure coordinates and sizes precisely (calculate by counting grid squares).
3. Convert all coordinates to be relative to R1.
4. Strictly adhere to the JSON format.
5. **Output ONLY the JSON, with no other text included.**

Now, analyze the new drawing and generate the JSON.
"""
    return prompt

def generate_json_with_gemini(image: Image.Image, prompt: str):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        
        response = model.generate_content(
            [prompt, image],
            generation_config={"temperature": 0.1, "candidate_count": 1},
            request_options={'timeout': 180}
        )

        json_text = response.text.strip()
        if '```json' in json_text:
            json_text = json_text.split('```json')[1].split('```')[0]
        elif '```' in json_text:
            json_text = json_text.split('```')[1].split('```')[0]
        
        return json.loads(json_text.strip())

    except KeyError:
        st.error("Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        return None
    except json.JSONDecodeError:
        st.error("AI가 유효한 JSON을 반환하지 않았습니다. 원본 응답을 확인하세요.")
        st.code(response.text, language='text')
        return None
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류 발생: {e}")
        return None

with st.sidebar:
    st.header("사용 방법")
    st.info("""
    1. 분석할 공장 도면 이미지 파일(PNG, JPG)을 업로드하세요.
    2. '분석 시작' 버튼을 누르면 AI가 이미지를 분석하여 오른쪽에 JSON 결과를 출력합니다.
    """)
    uploaded_file = st.file_uploader("여기에 도면 이미지 파일을 드래그하세요", type=['png', 'jpg', 'jpeg'])
    analyze_button = st.button("분석 시작", type="primary", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("CAD 도면 입력")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="분석 대상 이미지", use_container_width=True)
    else:
        st.info("시연을 시작하려면 왼쪽에서 도면 이미지를 업로드하세요.")

with col2:
    st.subheader("JSON 변환 결과")
    if uploaded_file and analyze_button:
        with st.spinner('AI가 도면을 분석하고 있습니다. 잠시만 기다려주세요...'):
            train_data = load_training_data("data/json_fixed.csv")
            
            if train_data:
                final_prompt = create_few_shot_prompt(train_data)
                result_json = generate_json_with_gemini(image, final_prompt)
                
                if result_json:
                    st.json(result_json)
                    st.success("분석이 성공적으로 완료되었습니다!")
    else:
        st.info("이미지를 업로드하고 '분석 시작' 버튼을 누르세요.")