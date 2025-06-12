import streamlit as st
from Rag import launch_depression_assistant, depression_assistant
# from openai import OpenAI
from together import Together

st.set_page_config(
    page_title="Depression Assistant Chatbot",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# you can add the embedder model to tried out more
# but it has to be from sentence-transformers library, or a encoder-only transformer model
model_options = [
    "Qwen/Qwen3-Embedding-0.6B",
    "jinaai/jina-embeddings-v3",
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "abhinand/MedEmbed-large-v0.1",
    "emilyalsentzer/Bio_ClinicalBERT",
    "Other"
]

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
        ["Default Free Together AI API","OpenAI", "Together AI", "NVIDIA", "Ollama"]
    )
    # Only show API key input if not using default free API
    if api_provider != "Default Free Together AI API" and api_provider != "Ollama":
        # Dynamic API key input based on selected provider
        api_key = st.text_input(f"{api_provider} API Key", type="password")
        if not api_key:
            st.info(f"Please add your {api_provider} API key to continue.", icon="🗝️")
    else:
        try:
            api_key = st.secrets["TOGETHER_API_KEY"]  # Replace with actual default key
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
    elif api_provider == "Together AI":
        llm_client = Together(api_key=together_api_key)
    elif api_provider == "NVIDIA":
        st.warning("NVIDIA entry is not yet tested in this app. Please use Together AI instead.")
        # llm_client = NvidiaLLM(nvidia_api_key)
        llm_client = None
    elif api_provider == "Ollama":
        llm_client = "Ollama"
        
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model for generation', ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free","Other"], key='selected_model')
    model_name = None
    if selected_model == "Other":
        #ask to input the model name
        model_name = st.sidebar.text_input('Enter the model name')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=100, max_value=1000, value=500, step=10)


# Show title and description.
st.title("💬 Depression Assistant Chatbot")


        
# --- Track launch state and embedder ---
if "launched" not in st.session_state:
    st.session_state.launched = False
if "assistant_launched" not in st.session_state:
    st.session_state.assistant_launched = False

# --- Launch button ---
if not st.session_state.launched:
    st.write("This is a simple depression assistant bot. You can ask questions related to depression and get responses based on clinical guidelines.")
    st.write(
    "This is a simple chatbot that uses **RAG (Retrieval-Augmented Generation)** to answer questions related to depression. "
    "It uses a combination of a retrieval model to find relevant documents and a language model to generate responses based on those documents."
    )
    st.write("You can choose different embedder models from the sidebar to see how they affect the responses.")
    st.write("You can also choose different API providers from the side bar to use different models and see how they affect the responses.")
    st.write("Please click the button below to launch the assistant. You'll have to reload the page to change the embedder model or API provider.")

    if st.button("Launch Assistant"):
        st.session_state.launched = True
        st.rerun()

# --- After launch ---
if st.session_state.launched:
    if not st.session_state.assistant_launched:
        launch_depression_assistant(embedder_name=embedder_name,designated_client=llm_client)
        st.session_state.assistant_launched = True

    # Initialize chat history
    # if "messages" not in st.session_state:
        # st.session_state.messages = []
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    

    # React to user input
    if prompt := st.chat_input("Ask me questions about the CANMAT depression guideline!"):
        # results, response = depression_assistant(prompt)
        # Divide the web page into two parts: left for results, right for response
        
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        placeholder = st.chat_message("assistant").empty()
        collected_text = ""
        if selected_model =="Other" and model_name is not None:
            results, response = depression_assistant(prompt, True, max_tokens=max_length, temperature=temperature, top_p=top_p, model_name=model_name)
        else:
            results, response = depression_assistant(prompt, True, max_tokens=max_length, temperature=temperature, top_p=top_p)

        for chunk in response:
            collected_text += chunk
            placeholder.markdown(collected_text)
    
        #add a button, if click on this button, split the page into half, left for results, right for response
        st.session_state.messages.append({"role": "assistant", "content": collected_text})
