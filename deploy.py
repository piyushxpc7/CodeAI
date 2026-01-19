
import os
import sys
import uuid
import re
import asyncio
import io
import time
from typing import List, Optional, Any, Dict, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import streamlit as st
import numpy as np

# Try importing external libraries (with fallback/instructions if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext
except ImportError:
    pass # Will be handled by requirements.txt

try:
    from PyPDF2 import PdfReader
    import faiss
    from sentence_transformers import SentenceTransformer
    from langchain_core.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOpenAI
    from huggingface_hub import InferenceClient
except ImportError:
    pass


# ==========================================
# 1. MODELS & SCHEMAS
# ==========================================

class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"

class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action" 
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"

class AgentRole(str, Enum):
    CODE_AGENT = "code_agent"
    ANALYSIS_AGENT = "analysis_agent"
    RAG_AGENT = "rag_agent"
    ASSISTANT = "assistant"

class AgentStep(BaseModel):
    step_number: int
    step_type: StepType
    content: str
    tool_used: Optional[str] = None
    agent_role: AgentRole
    timestamp: datetime = Field(default_factory=datetime.now)

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: Language = Language.PYTHON
    include_tests: bool = False
    include_docs: bool = True

class CodeGenerationResponse(BaseModel):
    code: str
    language: Language
    filename: Optional[str] = None
    explanation: Optional[str] = None
    dependencies: List[str] = []

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Language
    focus_areas: List[str] = ["security", "performance", "best_practices"]

class CodeAnalysisResponse(BaseModel):
    summary: str
    quality_score: int = Field(ge=0, le=100)
    strengths: List[str] = []
    recommendations: List[str] = []
    
class CodeSnippet(BaseModel):
    code: str
    language: str = "python"
    explanation: Optional[str] = None

class AssistantResponse(BaseModel):
    message: str
    code_snippets: List[CodeSnippet] = []
    suggestions: List[str] = []

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    page_number: int
    embedding: Optional[List[float]] = None

class RAGResponse(BaseModel):
    answer: str
    sources: List[DocumentChunk] = []
    confidence: float = 0.0

# ==========================================
# 2. UTILS & TOOLS
# ==========================================

class CodeTools:
    EXTENSIONS = {
        Language.PYTHON: ".py", Language.JAVASCRIPT: ".js", Language.TYPESCRIPT: ".ts",
        Language.JAVA: ".java", Language.CPP: ".cpp", Language.GO: ".go",
        Language.RUST: ".rs", Language.SQL: ".sql", Language.HTML: ".html", Language.CSS: ".css",
    }
    
    @staticmethod
    def get_extension(language: Language) -> str:
        return CodeTools.EXTENSIONS.get(language, ".txt")
    
    @staticmethod
    def suggest_filename(description: str, language: Language) -> str:
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        stop_words = {'a', 'an', 'the', 'to', 'for', 'of', 'in', 'create', 'write'}
        words = [w for w in words if w not in stop_words][:3]
        name = '_'.join(words) if words else "generated_code"
        return f"{name}{CodeTools.get_extension(language)}"

class PDFParser:
    def parse_pdf_bytes(self, file_bytes: bytes, doc_id: str) -> Tuple[List[DocumentChunk], int]:
        reader = PdfReader(io.BytesIO(file_bytes))
        chunks = []
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip(): continue
            
            # Simple chunking by paragraph
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if len(para) < 50: continue
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=para.strip(),
                    page_number=i+1
                ))
        return chunks, total_pages

class VectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.index = None
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunk_ids: List[str] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks: return
        texts = [c.content for c in chunks]
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()
            self.chunks[chunk.chunk_id] = chunk
            self.chunk_ids.append(chunk.chunk_id)

    def search(self, query: str, top_k: int = 5, document_filter: Optional[List[str]] = None) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or self.index.ntotal == 0: return []
        
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb).astype('float32'), top_k * 2) # Search more to filter
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks[chunk_id]
            
            if document_filter and chunk.doc_id not in document_filter:
                continue
                
            score = float(D[0][i])
            results.append((chunk, score))
            if len(results) >= top_k: break
            
        return results

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, Dict] = {} # doc_id -> metadata
        
    def add_document(self, filename: str, content: bytes, pages: int, chunks: int) -> str:
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "pages": pages,
            "chunks": chunks,
            "added_at": datetime.now()
        }
        return doc_id
        
    def list_documents(self) -> List[Dict]:
        return list(self.documents.values())

# ==========================================
# 3. AGENTS
# ==========================================

def get_model_provider(model_name: str) -> str:
    if model_name.startswith("hf/"): return "huggingface"
    return "openrouter"

class BaseAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = get_model_provider(model_name)
        self.steps = []
        
        if self.provider == "huggingface":
            self.hf_client = InferenceClient(token=os.getenv("HUGGINGFACE_API_KEY"))
            self.hf_model = model_name.replace("hf/", "")
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1"
            )

    async def _call_llm(self, messages: List[Dict]) -> str:
        if self.provider == "huggingface":
            def _call():
                return self.hf_client.chat.completions.create(
                    model=self.hf_model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7
                ).choices[0].message.content
            return await asyncio.get_event_loop().run_in_executor(None, _call)
        else:
            # LangChain for OpenRouter
            prompt = messages[-1]["content"] if messages else ""
            res = await self.llm.ainvoke(prompt)
            return res.content

class CodeGenerationAgent(BaseAgent):
    async def generate(self, req: CodeGenerationRequest) -> Tuple[CodeGenerationResponse, List[AgentStep]]:
        prompt = f"""
        Generate {req.language.value} code for: {req.prompt}
        Include tests: {req.include_tests}
        Include docs: {req.include_docs}
        
        Return ONLY the code block enclosed in ```{req.language.value}
        Then provide a brief explanation.
        """
        
        response_text = await self._call_llm([{"role": "user", "content": prompt}])
        
        # Parse Code
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response_text, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else response_text
        explanation = response_text.replace(f"```{req.language.value}", "").replace(code, "").replace("```", "").strip()
        
        filename = CodeTools.suggest_filename(req.prompt, req.language)
        
        return CodeGenerationResponse(
            code=code,
            language=req.language,
            filename=filename,
            explanation=explanation[:500] + "..." if len(explanation) > 500 else explanation
        ), []

class CodeAnalysisAgent(BaseAgent):
    async def analyze(self, req: CodeAnalysisRequest) -> Tuple[CodeAnalysisResponse, List[AgentStep]]:
        prompt = f"""
        Analyze this {req.language.value} code for {', '.join(req.focus_areas)}:
        
        ```
        {req.code}
        ```
        
        Provide:
        1. Quality Score (0-100)
        2. Strengths (bullet points)
        3. Recommendations (bullet points)
        4. Summary
        """
        
        txt = await self._call_llm([{"role": "user", "content": prompt}])
        
        # Simple parsing (robustness can be improved)
        score_match = re.search(r'Score:\s*(\d+)', txt)
        score = int(score_match.group(1)) if score_match else 75
        
        return CodeAnalysisResponse(
            summary=txt[:200] + "...",
            quality_score=score,
            strengths=["See full report"],
            recommendations=["See full report"]
        ), []

class CodeAssistant(BaseAgent):
    async def chat(self, message: str) -> AssistantResponse:
        txt = await self._call_llm([
            {"role": "system", "content": "You are a helpful coding assistant skilled in Python, JS, and software design."},
            {"role": "user", "content": message}
        ])
        
        snippets = []
        blocks = re.findall(r'```(\w+)?\n(.*?)```', txt, re.DOTALL)
        for lang, code in blocks:
            snippets.append(CodeSnippet(code=code.strip(), language=lang or "text"))
            
        return AssistantResponse(
            message=txt,
            code_snippets=snippets
        )

class RAGAgent(BaseAgent):
    def __init__(self, model_name, vector_store):
        super().__init__(model_name)
        self.vector_store = vector_store

    async def answer(self, query: str) -> Tuple[RAGResponse, List[AgentStep]]:
        results = self.vector_store.search(query, top_k=3)
        if not results:
            return RAGResponse(answer="No relevant documents found. Please upload a PDF."), []
            
        context = "\n\n".join([f"Source (Page {c.page_number}): {c.content}" for c, _ in results])
        
        prompt = f"""
        Answer the question based ONLY on the context below.
        
        Context:
        {context}
        
        Question: {query}
        """
        
        ans = await self._call_llm([{"role": "user", "content": prompt}])
        return RAGResponse(answer=ans, sources=[r[0] for r in results]), []


# ==========================================
# 4. STREAMLIT UI
# ==========================================

# Page Config
st.set_page_config(
    page_title="AgentCoder AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODELS = [
    ("🧠 Qwen 2.5 Coder", "hf/Qwen/Qwen2.5-Coder-32B-Instruct", "HuggingFace"),
    ("🤖 Llama 3.1 8B", "hf/meta-llama/Llama-3.1-8B-Instruct", "HuggingFace"),
    ("🦙 Llama 3.3 70B", "meta-llama/llama-3.3-70b-instruct:free", "OpenRouter"),
]
LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust"]

# Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "doc_store" not in st.session_state:
    st.session_state.doc_store = DocumentStore()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODELS[0][1]

# Helper
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.title("🚀 AgentCoder AI")
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        c1.metric("📄 Docs", len(st.session_state.doc_store.list_documents()))
        c2.metric("📦 Chunks", st.session_state.vector_store.total_chunks)
        
        st.markdown("### ⚙️ Model")
        opts = [m[0] for m in MODELS]
        model_map = {m[0]: m[1] for m in MODELS}
        sel = st.selectbox("Select Model", opts, index=0)
        st.session_state.selected_model = model_map[sel]
        
        if "hf/" in st.session_state.selected_model:
            st.success("✅ Free HuggingFace")
            if not os.getenv("HUGGINGFACE_API_KEY"):
                st.error("⚠️ HUGGINGFACE_API_KEY missing!")
        
        st.markdown("### 📚 Knowledge Base")
        uploaded = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded and st.button("Process PDF", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    parser = PDFParser()
                    chunks, pages = parser.parse_pdf_bytes(uploaded.getvalue(), str(uuid.uuid4()))
                    st.session_state.doc_store.add_document(uploaded.name, uploaded.getvalue(), pages, len(chunks))
                    st.session_state.vector_store.add_chunks(chunks)
                    st.success(f"Added {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"Error: {e}")

# Welcome
def render_welcome():
    if "welcomed" not in st.session_state:
        st.markdown("# 👋 Welcome to AgentCoder AI")
        st.markdown("Your AI-powered coding companion.")
        if st.button("Get Started 🚀", type="primary"):
            st.session_state.welcomed = True
            st.rerun()
        return True
    return False

# Tabs
def render_generate():
    st.header("💻 Code Generation")
    c1, c2 = st.columns([3, 1])
    with c1: prompt = st.text_area("What to build?", height=150)
    with c2: 
        lang = st.selectbox("Language", LANGUAGES)
        tests = st.toggle("Tests", False)
        docs = st.toggle("Docs", True)
        
    if st.button("Generate", type="primary", disabled=not prompt):
        with st.status("Generating...") as status:
            try:
                agent = CodeGenerationAgent(st.session_state.selected_model)
                req = CodeGenerationRequest(prompt=prompt, language=Language(lang.lower()), include_tests=tests, include_docs=docs)
                res, _ = run_async(agent.generate(req))
                status.update(label="✅ Done!", state="complete")
                
                st.subheader(f"📝 {res.filename}")
                st.code(res.code, language=res.language.value)
                st.download_button("Download", res.code, file_name=res.filename)
                if res.explanation: st.info(res.explanation)
            except Exception as e:
                status.update(label="❌ Failed", state="error")
                st.error(str(e))

def render_analyze():
    st.header("🔍 Code Analysis")
    col1, col2 = st.columns([3, 1])
    with col1: code = st.text_area("Paste code", height=200)
    with col2: lang = st.selectbox("Lang", LANGUAGES, key="aly")
    
    if st.button("Analyze", type="primary", disabled=not code):
        with st.status("Analyzing...") as status:
            try:
                agent = CodeAnalysisAgent(st.session_state.selected_model)
                req = CodeAnalysisRequest(code=code, language=Language(lang.lower()), focus_areas=[])
                res, _ = run_async(agent.analyze(req))
                status.update(label="✅ Done!", state="complete")
                
                st.metric("Quality Score", f"{res.quality_score}/100")
                st.progress(res.quality_score/100)
                st.info(res.summary)
            except Exception as e:
                status.update(label="❌ Failed", state="error")
                st.error(str(e))

def render_assistant():
    st.header("🤖 Assistant")
    if "chat_mx" not in st.session_state: st.session_state.chat_mx = []
    
    for m in st.session_state.chat_mx:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            
    if p := st.chat_input("Ask help..."):
        st.session_state.chat_mx.append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = CodeAssistant(st.session_state.selected_model)
                res = run_async(agent.chat(p))
                st.write(res.message)
                for s in res.code_snippets: st.code(s.code, language=s.language)
                st.session_state.chat_mx.append({"role": "assistant", "content": res.message})

def render_rag():
    st.header("📚 Doc Q&A")
    if not st.session_state.doc_store.list_documents():
        st.info("Upload PDF first")
        return
        
    if "rag_mx" not in st.session_state: st.session_state.rag_mx = []
    
    for m in st.session_state.rag_mx:
        with st.chat_message(m["role"]): st.write(m["content"])
        
    if p := st.chat_input("Ask docs..."):
        st.session_state.rag_mx.append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                agent = RAGAgent(st.session_state.selected_model, st.session_state.vector_store)
                res, _ = run_async(agent.answer(p))
                st.write(res.answer)
                st.session_state.rag_mx.append({"role": "assistant", "content": res.answer})

def main():
    render_sidebar()
    if render_welcome(): return
    t1, t2, t3, t4 = st.tabs(["💻 Generate", "🔍 Analyze", "🤖 Assistant", "📚 Docs"])
    with t1: render_generate()
    with t2: render_analyze()
    with t3: render_assistant()
    with t4: render_rag()

if __name__ == "__main__":
    main()
