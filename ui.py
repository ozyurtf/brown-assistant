import os
import requests
import streamlit as st
from utils import map_code_to_dept_cab, map_code_to_dept_bulletin

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

bulletin_code_to_dept_map = map_code_to_dept_bulletin()
cab_code_to_dept_map = map_code_to_dept_cab()

st.set_page_config(page_title="RAG UI", layout="wide")
st.title("Student Course Assistant")

with st.sidebar:
    st.header("Filters")
    
    bulletin_opts = ["Any"] + list(bulletin_code_to_dept_map.keys())
    bulletin_department = st.selectbox("Bulletin department", bulletin_opts, index=0)

    cab_opts = ["Any"] + list(cab_code_to_dept_map.keys())
    cab_department = st.selectbox("CAB Department", cab_opts, index=0)

question = st.text_area("Ask a question", placeholder="e.g., Does APMA 1650 satisfy my requirements?", height=120)

if st.button("Search", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        b_dept = None if bulletin_department == "Any" else bulletin_code_to_dept_map[bulletin_department].lower()
        c_dept = None if cab_department == "Any" else cab_code_to_dept_map[cab_department]
        if not b_dept and not c_dept:
            st.warning("Select at least one department (bulletin or CAB).")
        print("Selected departments:")
        print(b_dept, c_dept)
        print()

        payload = {
            "question": question.strip(),
            "bulletin_department": b_dept,
            "cab_department": c_dept,
            "top_k_bulletin": 3,
            "top_k_cab": 5,
        }
        try:
            with st.spinner("Retrieving and generating answer..."):
                response = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            st.subheader("Retrieved Context")
            for i, item in enumerate(data.get("retrieved", []), start=1):
                meta = item.get("metadata", {}) or {}
                source = meta.get('source','')
                department_code = meta.get('department','')
                if source == 'bulletin':
                    source = 'Bulletin'
                    reversed_dict = {v: k for k, v in bulletin_code_to_dept_map.items()}
                    department = reversed_dict[department_code]
                elif source == 'cab':
                    source = 'CAB'
                    reversed_dict = {v: k for k, v in cab_code_to_dept_map.items()}
                    department = reversed_dict[department_code]
                with st.expander(f"{i}) Context — {source} ({department})"):
                    st.write(item.get("text", ""))
