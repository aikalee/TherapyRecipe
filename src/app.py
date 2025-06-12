import streamlit as st
from Rag import launch_depression_assistant, depression_assistant
# from openai import OpenAI
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

    # 显示历史
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 用户输入
    if user_input := st.chat_input("Ask me questions about the CANMAT depression guideline!"):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # ===== 取最近 4 轮历史 =====
        history = st.session_state.messages[:-1][-4:]

        placeholder = st.chat_message("assistant").empty()
        collected = ""

        for chunk in depression_assistant(user_input, True, chat_history=history):
            collected += chunk
            placeholder.markdown(collected)

        st.session_state.messages.append({"role": "assistant", "content": collected})
