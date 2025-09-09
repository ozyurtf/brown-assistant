import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from utils import map_code_to_dept, map_code_to_concentration

load_dotenv()

# Determine API base URL with sensible fallbacks

host = os.getenv("API_HOST")
port = os.getenv("API_PORT")
if host and port:
    API_BASE = f"http://{host}:{port}"
else:
    API_BASE = "http://localhost:8000"

# Get API token for authentication
API_TOKEN = os.getenv("API_TOKEN")

# If API_TOKEN is still empty, manually parse the .env file
if not API_TOKEN:
    try:
        with open(".env", 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('API_TOKEN='):
                    API_TOKEN = line.split('=', 1)[1]
                    break
    except Exception as e:
        print(f"Error reading .env file: {e}")

REQUEST_TIMEOUT = 180

def get_auth_headers():
    """Get authorization headers if API token is configured."""
    if API_TOKEN:
        return {"Authorization": f"Bearer {API_TOKEN}"}
    return {}  

@st.cache_data(show_spinner=False)
def fetch_evaluation(embedding_model: str):
    """Fetch evaluation data for specified embedding model."""
    try:
        response = requests.post(
            f"{API_BASE}/evaluate", 
            json={"embedding_model": embedding_model}, 
            headers=get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def create_model_display_name(model_key: str, model_value: str) -> str:
    """Create a user-friendly display name for the model."""
    model_name_map = {
        "sentence_transformer": "Sentence Transformer",
        "openai": "OpenAI"
    }
    base_name = model_name_map.get(model_key, model_key.replace("_", " ").title())
    return f"{base_name} ({model_value})"

# Initialize department mappings
code_to_concentration_map = map_code_to_concentration()
code_to_department_map = map_code_to_dept()

st.set_page_config(page_title="Brown Assisstant", layout="wide")
st.title("Brown Assisstant")

with st.sidebar:
    st.header("Model Selection")
    
    # Fetch available models
    available_models = ["sentence_transformer", "openai"]
    model_details = {
        "sentence_transformer": "all-MiniLM-L6-v2",
        "openai": "text-embedding-3-large"
    }
    default_model = "sentence_transformer"
    
    # Create model options with display names
    model_options = {}
    for model_key in available_models:
        if model_key in model_details:
            model_options[model_key] = create_model_display_name(model_key, model_details[model_key])
        else:
            model_options[model_key] = model_key.replace("_", " ").title()
    
    # Model selection with proper default
    try:
        default_index = available_models.index(default_model)
    except ValueError:
        default_index = 0
    
    selected_embedding_model = st.selectbox(
        "Choose Embedding Model",
        options=available_models,
        index=default_index,
        format_func=lambda x: model_options.get(x, x),
        help="Select which embedding model to use for retrieval"
    )
    
    st.success("Connected to API")
    st.divider()
    st.header("Filters")
    
    # Add informational note about data sources
    with st.expander("When to filter by department vs concentration?", expanded=False):
        st.warning("""
**Department** - Use for current course offerings and semester planning:
• What courses are available this semester  
• Course schedules, meeting times, instructors  
• Specific course descriptions and prerequisites  
*Example: "What CSCI courses are offered this fall?" or "When does APMA 1650 meet?"*

**Concentration** - Use for degree requirements and academic policies:
• What courses are required for your concentration  
• Official graduation requirements and academic rules  
• Concentration policies, honors eligibility, capstone requirements  
*Example: "What are the requirements for Computer Science concentration?"*

**Tip:** Select both filters to get comprehensive information combining requirements with current offerings
        """)
    
    dept_opts = [None] + list(code_to_department_map.keys())
    dept_selectbox = st.selectbox(
        "Department", 
        dept_opts, 
        index=0,
        help="Filter by department for current course offerings and schedules"
    )

    concentration_opts = [None] + list(code_to_concentration_map.keys())
    concentration_selectbox = st.selectbox(
        "Concentration", 
        concentration_opts, 
        index=0,
        help="Filter by concentration/major for degree requirements and policies"
    )

    term_opts = ["Fall 2025"]
    term_selectbox = st.selectbox(
        "Term", 
        term_opts, 
        index=0,
        help="Select the term for which you want to search for courses"
    )    

    

# Main query interface
question = st.text_area(
    "Ask a question", 
    placeholder="e.g., Does APMA 1650 satisfy my requirements?", 
    height=120
)

left_col, right_col, _spacer = st.columns([0.12, 0.12, 0.76], gap="small")

with left_col:
    search_clicked = st.button("Search", type="primary")
with right_col:
    evaluate_clicked = st.button("Evaluate")

if search_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Process department selections
        conc = None if concentration_selectbox == None else code_to_concentration_map[concentration_selectbox].lower()
        dept = None if dept_selectbox == None else code_to_department_map[dept_selectbox]
        
        if not conc and not dept:
            st.warning("Select a concentration and/or department.")
        else:
            # Prepare request payload
            payload = {
                "question": question.strip(),
                "concentration": conc,
                "department": dept, 
                "rerank_model_name": "BAAI/bge-reranker-base",
                "embedding_model": selected_embedding_model,
            }
            
            try:
                retrieval_time_seconds = None
                current_model_display = model_options.get(selected_embedding_model, '')
                with st.spinner(f"Retrieving relevant information with {current_model_display} and generating an answer."):
                    start_time = time.perf_counter()
                    response = requests.post(
                        f"{API_BASE}/query", 
                        json=payload, 
                        headers=get_auth_headers(),
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()
                    retrieval_time_seconds = time.perf_counter() - start_time
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Please check if the server is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error ({e.response.status_code}): {e.response.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
                
            else:
                # Display results
                st.subheader("Answer")
                answer = data.get("answer", "No answer provided")
                st.write(answer)
                
                # Show which model was used
                st.info(f"Retrieved using: {current_model_display}")
                if retrieval_time_seconds is not None:
                    st.subheader("Retrieval Time")
                    st.write(f"Retrieval Time: **{retrieval_time_seconds:.2f} s**")

                # Display retrieved context
                retrieved_results = data.get("retrieved", [])
                if retrieved_results:
                    st.subheader("Retrieved Context")
                    
                    for i, item in enumerate(retrieved_results, start=1):
                        meta = item.get("metadata", {}) or {}
                        source = meta.get('source', 'unknown')
                        department_code = meta.get('department', 'unknown')
                        similarity_score = item.get('similarity_score', 0.0)
                        embedding_model_used = meta.get('embedding_model', 'unknown')
                        
                        # Format source and department display
                        if source.lower() == 'bulletin':
                            source_display = 'Bulletin'
                            reversed_dict = {v: k for k, v in code_to_concentration_map.items()}
                            department_display = reversed_dict.get(department_code, department_code)
                        elif source.lower() == 'cab':
                            source_display = 'CAB'
                            reversed_dict = {v: k for k, v in code_to_department_map.items()}
                            department_display = reversed_dict.get(department_code, department_code)
                        else:
                            source_display = source.title()
                            department_display = department_code
                        
                        # Format similarity score
                        try:
                            similarity_display = f"{float(similarity_score):.4f}"
                        except (ValueError, TypeError):
                            similarity_display = "N/A"
                        
                        # Create expander for each context item
                        expander_title = f"{i}) Context — {source_display} ({department_display}) (Similarity: {similarity_display})"
                        
                        with st.expander(expander_title):
                            context_text = item.get("text", "No text available")
                            st.write(context_text)
                            st.caption(f"Embedding model: {embedding_model_used}")
                else:
                    st.warning("No context was retrieved for this query.")

if evaluate_clicked:
    current_model_display = None
    try:
        current_model_display = model_options.get(selected_embedding_model, selected_embedding_model)
        with st.spinner(f"Evaluating {current_model_display}"):
            eval_data = fetch_evaluation(selected_embedding_model)
    except Exception as e:
        eval_data = {"error": str(e)}

    if eval_data and not eval_data.get("error"):
        def format_score(score):
            try:
                return f"{float(score):.3f}"
            except (ValueError, TypeError):
                return "N/A"

        bleu_score = format_score(eval_data.get('bleu_score', 0.0))
        rouge_score = format_score(eval_data.get('rouge_score', 0.0))
        
        st.subheader("Evaluation Metrics")
        st.write(f"Model: **{current_model_display or selected_embedding_model}**")
        m1, m2= st.columns(2)
        with m1:
            st.metric("BLEU Score", bleu_score)
        with m2:
            st.metric("ROUGE-L Score", rouge_score)
        st.caption("Averages computed on evaluation.json")
    else:
        st.error(f"Evaluation not available for {current_model_display or selected_embedding_model}")
        if eval_data.get("error"):
            st.caption(f"Error: {eval_data['error']}")  