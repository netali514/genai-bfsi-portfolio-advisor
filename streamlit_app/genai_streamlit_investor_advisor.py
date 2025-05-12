import streamlit as st
import subprocess
import time
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def load_vectorstore():
    with open("finance_knowledge.txt", "w") as f:
        f.write("""Gold is a hedge against inflation and market uncertainty.
Equities offer higher returns but come with increased risk.
Fixed income instruments like bonds provide stable income with lower risk.
Diversification reduces risk in an investment portfolio.
Young investors can take more risk than older investors.
Mutual funds are ideal for beginners due to diversification and professional management.
""")

    local_docs = TextLoader("finance_knowledge.txt").load()
    url = "https://www.investopedia.com/articles/basics/06/invest1000.asp"
    web_docs = WebBaseLoader(url).load()
    docs = local_docs + web_docs

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [doc.page_content for doc in chunks]

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embedder)

def is_response_safe(text):
    blocked = ["guaranteed returns", "legal advice", "tax evasion", "insider information", "buy this stock"]
    return not any(term in text.lower() for term in blocked)

def evaluate(text):
    return {
        "relevance": int(any(w in text.lower() for w in ["gold", "bonds", "equity", "diversification"])),
        "clarity": int("%" in text or "allocation" in text.lower()),
        "safety": int(is_response_safe(text))
    }

def make_prompt(goal, age, risk, amount, duration, context):
    return f"""Context:
{context}

Investor Details:
- Goal: {goal}
- Age: {age}
- Risk Appetite: {risk}
- Investment Amount: {amount}
- Investment Duration: {duration} years

Question:
Given this profile, what asset allocation and investment strategy would you recommend? Please justify each allocation.
"""

def get_response(prompt):
    start = time.time()
    proc = subprocess.Popen(["ollama", "run", "llama3"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    output, _ = proc.communicate(prompt)
    return output.strip(), time.time() - start

st.title("💰 GenAI Investment Advisor (Free + Local)")
st.markdown("Enter your investment profile below to receive personalized strategy advice.")

goal = st.text_input("🎯 Investment Goal", "Save for child's education")
age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
risk = st.selectbox("⚖️ Risk Appetite", ["Low", "Moderate", "High"])
amount = st.text_input("💵 Amount to Invest", "$20,000")
duration = st.slider("📅 Investment Duration (Years)", 1, 30, 10)

if st.button("🔍 Get Investment Strategy"):
    with st.spinner("Thinking..."):
        vectorstore = load_vectorstore()
        query = f"{goal} investment plan"
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = make_prompt(goal, age, risk, amount, duration, context)
        reply, latency = get_response(prompt)
        score = evaluate(reply)

    st.subheader("🧠 AI-Generated Advice")
    st.code(reply, language="markdown")

    st.subheader("📊 Evaluation")
    st.write(f"**Relevance**: {score['relevance']}/1")
    st.write(f"**Clarity**: {score['clarity']}/1")
    st.write(f"**Safety**: {score['safety']}/1")
    st.write(f"⏱️ Response Time: {latency:.2f} seconds")
