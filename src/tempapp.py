import streamlit as st
from Rag import launch_depression_assistant, depression_assistant
from openai import OpenAI
from together import Together
from dotenv import load_dotenv

# ---------- Sidebar ----------
st.sidebar.title("Settings")

model_options = [
    "all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v3",
    "paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "abhinand/MedEmbed-large-v0.1"
]
embedder_name = st.sidebar.selectbox("Select embedder model", model_options, 0)

# --- Sidebar ---
st.sidebar.title("Settings")
with st.sidebar:

    st.subheader("Embedder and LLM API")
    embedder_name = st.sidebar.selectbox(
    "Select embedder model",
    model_options,
    index=0
    )
    if embedder_name == "Other":
        # Ask to input the embedder model name
        embedder_name = st.sidebar.text_input('Enter the embedder model name')
    # API provider selection
    api_provider = st.sidebar.selectbox(
        "Select API Provider",
        ["Default Free Nvidia API","Default Free Together AI API","OpenAI", "Together AI", "NVIDIA", "Run Ollama Locally"]
    )
    # Only show API key input if not using default free API
    if api_provider not in ["Default Free Nvidia API","Default Free Together AI API", "Run Ollama Locally"]:
        # Dynamic API key input based on selected provider
        api_key = st.text_input(f"{api_provider} API Key", type="password")
        if not api_key:
            st.info(f"Please add your {api_provider} API key to continue.", icon="🗝️")
    else:
        try:
            if api_provider == "Default Free Nvidia API":
                api_key = st.secrets["NVIDIA_API_KEY"]
                llm_client = OpenAI(
                    base_url = "https://integrate.api.nvidia.com/v1",
                    api_key = api_key,
                )
                print("NVIDIA client initialized with default key")
            elif api_provider == "Default Free Together AI API":
                api_key = st.secrets["TOGETHER_API_KEY"]
                llm_client =Together(api_key=api_key)
        except Exception as e:
            st.warning("Default API key not found. Probably running locally")
            llm_client = None
    openai_api_key = api_key if api_provider == "OpenAI" else None
    together_api_key = api_key if api_provider == "Together AI" else None
    nvidia_api_key = api_key if api_provider == "NVIDIA" else None
    
    # Initialize LLM client based on provider
    if api_provider == "OpenAI":
        if openai_api_key:
            llm_client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized")
    elif api_provider == "Together AI":
        llm_client = Together(api_key=together_api_key)
        print("Together AI client initialized")
    elif api_provider == "NVIDIA":
        if nvidia_api_key:
            llm_client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = nvidia_api_key,
            )
            print("NVIDIA client initialized")
        else:
            st.warning("NVIDIA API key is required to use NVIDIA models.")
            llm_client = None
    elif api_provider == "Run Ollama Locally":
        llm_client = "Run Ollama Locally"
        print("Select to run Ollama client. Please start the Ollama server locally and make sure the model is downloaded.")
        
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model for generation', ["deepseek-ai/deepseek-r1","meta/llama-3.3-70b-instruct","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free","Other"], key='selected_model')
    model_name = None
    if selected_model == "Other":
        #ask to input the model name
        model_name = st.sidebar.text_input('Enter the model name')
    print(f"Selected model: {selected_model}, Model name: {model_name}")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=100, max_value=1000, value=500, step=10)


# Show title and description.
st.title("💬 Depression Assistant Chatbot")

api_provider = st.sidebar.selectbox("Select API Provider",
                                    ["Default Free Together AI API", "OpenAI", "Together AI", "NVIDIA"])

if api_provider != "Default Free Together AI API":
    api_key = st.text_input(f"{api_provider} API Key", type="password")
    if not api_key:
        st.info(f"Please add your {api_provider} API key to continue.", icon="🗝️")
else:
    try:
        api_key = st.secrets["TOGETHER_API_KEY"]
        llm_client = Together(api_key=api_key)
    except Exception:
        st.warning("Default API key not found. Probably running locally")
        llm_client = None

if api_provider == "OpenAI":
    from openai import OpenAI
    llm_client = OpenAI(api_key=api_key)
elif api_provider == "Together AI":
    llm_client = Together(api_key)
elif api_provider == "NVIDIA":
    st.warning("NVIDIA not yet supported; using Together free tier")
    llm_client = None

# ---------- Session state ----------
if "launched" not in st.session_state:
    st.session_state.launched = False
if "assistant_launched" not in st.session_state:
    st.session_state.assistant_launched = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Launch button ----------
if not st.session_state.launched:
    st.markdown(
        "This chatbot uses **RAG** to answer depression-related questions based on clinical guidelines.\n\n"
        "Choose an embedder/API on the left, then click **Launch Assistant**."
    )
    if st.button("Launch Assistant"):
        st.session_state.launched = True
        st.rerun()

# ---------- After launch ----------
if st.session_state.launched:
    if not st.session_state.assistant_launched:
        launch_depression_assistant(embedder_name=embedder_name, designated_client=llm_client)
        st.session_state.assistant_launched = True
        st.session_state.messages.append({"role": "assistant", "content": "How may I assist you today?"})

    # show histories
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # user input
    if user_input := st.chat_input("Ask me questions about the CANMAT depression guideline!"):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # ===== latest 4 histories =====
        history = st.session_state.messages[:-1][-4:]

        placeholder = st.chat_message("assistant").empty()
        collected = ""
        
        if selected_model =="Other" and model_name is not None:
            results, response = depression_assistant(user_input, model_name=model_name, max_tokens=max_length, temperature=temperature, top_p=top_p,  stream_flag=True, chat_history=history)
        else:
            results, response = depression_assistant(user_input, model_name=selected_model, max_tokens=max_length, temperature=temperature, top_p=top_p, model_name=selected_model, stream_flag=True, chat_history=history)

        for chunk in response:
            collected += chunk
            placeholder.markdown(collected)

        st.session_state.messages.append({"role": "assistant", "content": collected})
