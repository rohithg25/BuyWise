import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🛒 Electronics Shopping Assistant",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Electronics Shopping Assistant")
st.write(
    "Ask me anything about cameras and electronics — based ONLY on the dataset.\n\n"
    "Examples:\n"
    "- Best Canon DSLR camera\n"
    "- Sony cameras under 50000\n"
    "- Highest rated Nikon camera\n"
    "- Panasonic mirrorless cameras"
)

# =========================
# LLM + PROMPT
# =========================
@st.cache_resource
def load_chain():
    model = OllamaLLM(model="mistral")

    template = """
You are an electronics shopping assistant.

STRICT RULES:
- Use ONLY the provided product dataset
- Do NOT invent products, prices, or specifications
- Do NOT give general shopping advice
- Answer ONLY from the dataset
- If the product or information is not found, say:
  "This information is not available in the dataset."

Product dataset:
{context}

User question:
{question}
"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model


chain = load_chain()

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# USER INPUT
# =========================
question = st.chat_input("Ask about cameras or electronics...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching products... 🔍"):
            docs = retriever.invoke(question)

            if not docs:
                response = "This information is not available in the dataset."
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                response = chain.invoke({
                    "context": context,
                    "question": question
                })

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })