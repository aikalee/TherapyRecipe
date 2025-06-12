import streamlit as st
from Rag import launch_depression_assistant, depression_assistant
# from openai import OpenAI
from together import Together

# --- Sidebar ---
st.sidebar.title("Settings")
with st.sidebar:

    st.subheader("Embedder and LLM API")
    model_options = [
        "all-MiniLM-L6-v2",
        "jinaai/jina-embeddings-v3",
        "paraphrase-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "abhinand/MedEmbed-large-v0.1",
        "Qwen/Qwen3-Embedding-0.6B",
        "emilyalsentzer/Bio_ClinicalBERT",
        "Other"
    ]
    embedder_name = st.selectbox(
        "Select embedder model",
        model_options,
        index=0
    )
    if embedder_name == "Other":
        embedder_name = st.text_input("Enter the embedder model name")
    
    api_provider = st.selectbox(
        "Select API Provider",
        ["Default Free Together AI API", "OpenAI", "Together AI", "NVIDIA"]
    )
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

    openai_api_key = api_key if api_provider == "OpenAI" else None
    together_api_key = api_key if api_provider == "Together AI" else None
    nvidia_api_key = api_key if api_provider == "NVIDIA" else None

    if api_provider == "OpenAI" and openai_api_key:
        from openai import OpenAI
        llm_client = OpenAI(api_key=openai_api_key)
    elif api_provider == "Together AI" and together_api_key:
        llm_client = Together(api_key=together_api_key)
    elif api_provider == "NVIDIA":
        st.warning("NVIDIA entry is not yet tested in this app. Please use Together AI instead.")
        llm_client = None

    st.subheader("Models and parameters")
    selected_model = st.selectbox(
        "Choose a model for generation",
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Other"],
        key="selected_model"
    )
    model_name = None
    if selected_model == "Other":
        model_name = st.text_input("Enter the model name")
    temperature = st.slider("temperature", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    top_p = st.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider("max_length", min_value=100, max_value=1000, value=500, step=10)

# Show title and description.
st.title("💬 Depression Assistant Chatbot")

# --- Track launch state ---
if "launched" not in st.session_state:
    st.session_state.launched = False
if "assistant_launched" not in st.session_state:
    st.session_state.assistant_launched = False

# --- Launch button ---
if not st.session_state.launched:
    st.write(
        "This is a simple depression assistant bot. You can ask questions related to depression "
        "and get responses based on clinical guidelines."
    )
    st.write(
        "This chatbot uses **RAG (Retrieval-Augmented Generation)** under the hood. "
        "Pick your embedder and LLM provider on the left, then launch the assistant."
    )
    if st.button("Launch Assistant"):
        st.session_state.launched = True
        st.rerun()

# --- After launch ---
if st.session_state.launched:
    if not st.session_state.assistant_launched:
        launch_depression_assistant(embedder_name=embedder_name, designated_client=llm_client)
        st.session_state.assistant_launched = True

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me questions about the CANMAT depression guideline!"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # =====latest 4 history =====
        history = st.session_state.messages[:-1][-4:]

        placeholder = st.chat_message("assistant").empty()
        collected_text = ""

        # Call the assistant with memory
        if selected_model == "Other" and model_name:
            _, response = depression_assistant(
                prompt,
                True,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                model_name=model_name,
                chat_history=history
            )
        else:
            _, response = depression_assistant(
                prompt,
                True,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                chat_history=history
            )

        # Stream the response
        for chunk in response:
            collected_text += chunk
            placeholder.markdown(collected_text)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": collected_text})
