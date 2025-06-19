import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.llms.base import LLM
from typing import List
from groq import Groq
import os
#from dotenv import load_dotenv
#load_dotenv()

# ------------ CONFIGURATION PAGE ------------
st.set_page_config(page_title="Chatbot Juridique SN", page_icon="⚖️", layout="wide")

col1, col2 = st.columns([1, 8])
with col1:
    st.image("drapeau justicesn.jpg", width=150)
with col2:
    st.markdown("""
        <h1 style='margin-top: 25px; font-size: 36px; color: #003366;'>
            LexSN : Votre assistant juridique de recherche documentaire
        </h1>
    """, unsafe_allow_html=True)

st.divider()

# ------------ CLASSE PERSONNALISÉE GROQLLM ------------
class GroqLLM(LLM):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    api_key: str = st.secrets["GROQ_API_KEY"]

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

# ------------ CHARGEMENT CHAINE RAG ------------
@st.cache_resource
def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(persist_directory="chroma_base", embedding_function=embedding_model)

    prompt_template = PromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans les textes de loi du Sénégal (Code de la famille, Code pénal, décrets, lois, etc).

Ta mission est de répondre de manière claire, concise et fiable à des questions posées par un utilisateur en t'appuyant exclusivement sur les extraits de documents juridiques suivants :

{context}

Consignes strictes :

- Réponds uniquement à partir du contenu fourni dans les extraits ci-dessus.
- Ne fais aucune supposition ni déduction en dehors des textes.
- N'invente jamais de références, de lois, ni de liens.
- Si l’information n’est pas présente, dis simplement : « Je suis désolé, mais aucun extrait de document en ma possession ne semble contenir une réponse claire à cette question. »
- Utilise un ton neutre, factuel et professionnel mais des réponses longues et explicatives.
- Réponds dans la langue de la question posée : français ou anglais.

---

Question : {question}

Réponse :
    """)

    llm = GroqLLM()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="mmr", k=2),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

qa_chain = load_qa_chain()

# ------------ AFFICHAGE MESSAGES ------------
USER_ICON = ""
BOT_ICON = "⚖️"
if "messages" not in st.session_state:
    st.session_state.messages = []

def message_bulle(texte, role="user"):
    icon = USER_ICON if role == "user" else BOT_ICON
    bubble_color = "#DCF8C6" if role == "user" else "#E6E6E6"
    st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
            <div style='font-size: 30px; margin-right: 10px;'>{icon}</div>
            <div style='background-color:{bubble_color}; padding:15px; border-radius:12px; max-width: 80%; font-size: 22px;'>
                {texte}
            </div>
        </div>
    """, unsafe_allow_html=True)

for m in st.session_state.messages:
    message_bulle(m["content"], m["role"])

# ------------ INPUT UTILISATEUR ------------
st.markdown("""
    <style>
        .stTextInput>div>input {
            border-radius: 25px;
            padding: 16px;
            font-size: 22px;
            border: 1px solid #ccc;
        }
        button[kind="primary"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Pose ta question sur le droit sénégalais :", "", placeholder="Ex: Quels sont les droits des femmes dans le code de la famille ?")
    submit = st.form_submit_button("Envoyer")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Recherche juridique en cours..."):
        import time
        start = time.time()
        result = qa_chain.invoke(user_input)
        elapsed = time.time() - start

        reponse = result["result"]
        sources = result["source_documents"]

    st.session_state.messages.append({"role": "bot", "content": reponse})

    # Affichage des sources
    unique_seen = set()
    limited_sources_list = []

    for doc in sources:
        meta = doc.metadata
        source_id = meta.get("document_title", "") + meta.get("chunk_title", "")
        if source_id not in unique_seen:
            folder = meta.get("folder", "Sans sous-dossier")
            title = meta.get("chunk_title", "Sans titre")
            source = meta.get("document_title", "Inconnu")
            url = meta.get("source_url", "Source inconnue")
            label = f"📚 {folder}/{source} / {title}\n→ {url}"
            limited_sources_list.append(label)
            unique_seen.add(source_id)
        if len(limited_sources_list) == 2:
            break

    st.session_state.messages.append({
        "role": "bot",
        "content": "🔎 Sources utilisées :\n\n" + "\n\n".join(limited_sources_list)
    })

    st.session_state.messages.append({
        "role": "bot",
        "content": f"⏱️ Réponse générée en **{elapsed:.2f} secondes**"
    })

    st.rerun()
