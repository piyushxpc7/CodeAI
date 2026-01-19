
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

class DocumentChunk(BaseModel):
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
        return RAGResponse(answer=ans or "No answer generated.", sources=[r[0] for r in results]), []


# ==========================================
# 4. STREAMLIT UI - ENHANCED SILICON VALLEY STYLE
# ==========================================

# Page Config
st.set_page_config(
    page_title="AgentCoder AI | Your AI Coding Companion",
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
        st.markdown("# 🚀 AgentCoder AI")
        st.caption("AI-Powered Development Platform")
        st.divider()
        
        # Metrics in beautiful cards
        st.markdown("### 📊 Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Documents", 
                value=len(st.session_state.doc_store.list_documents()),
                delta="Active" if len(st.session_state.doc_store.list_documents()) > 0 else None
            )
        with col2:
            st.metric(
                label="Chunks", 
                value=st.session_state.vector_store.total_chunks,
                delta="+New" if st.session_state.vector_store.total_chunks > 0 else None
            )
        
        st.divider()
        
        # Model Selection
        st.markdown("### 🤖 AI Model")
        opts = [m[0] for m in MODELS]
        model_map = {m[0]: m[1] for m in MODELS}
        provider_map = {m[0]: m[2] for m in MODELS}
        
        sel = st.selectbox(
            "Choose your model",
            opts,
            index=0,
            help="Select the AI model for code generation and analysis"
        )
        st.session_state.selected_model = model_map[sel]
        
        # Model info
        provider = provider_map[sel]
        if "hf/" in st.session_state.selected_model:
            st.success(f"✅ {provider} (Free Tier)")
            if not os.getenv("HUGGINGFACE_API_KEY"):
                st.error("⚠️ API Key Required")
                st.caption("Add HUGGINGFACE_API_KEY to .env")
        else:
            st.info(f"ℹ️ Using {provider}")
        
        st.divider()
        
        # Knowledge Base Upload
        st.markdown("### 📚 Knowledge Base")
        st.caption("Upload PDFs to enable RAG-powered Q&A")
        
        uploaded = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload documentation for context-aware answers",
            label_visibility="collapsed"
        )
        
        if uploaded:
            if st.button("🔄 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing document..."):
                    try:
                        parser = PDFParser()
                        chunks, pages = parser.parse_pdf_bytes(uploaded.getvalue(), str(uuid.uuid4()))
                        st.session_state.doc_store.add_document(uploaded.name, uploaded.getvalue(), pages, len(chunks))
                        st.session_state.vector_store.add_chunks(chunks)
                        st.success(f"✅ Processed {pages} pages, {len(chunks)} chunks!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Footer
        st.caption("Made with ❤️ using Streamlit")
        st.caption("© 2024 AgentCoder AI")

# Welcome Screen
def render_welcome():
    if "welcomed" not in st.session_state:
        # Hero Section
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("# 🚀 Welcome to AgentCoder AI")
            st.markdown("### Your Intelligent Coding Companion")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Feature cards
            feat_col1, feat_col2, feat_col3 = st.columns(3)
            with feat_col1:
                st.info("**💻 Code Generation**\n\nGenerate production-ready code in multiple languages")
            with feat_col2:
                st.info("**🔍 Code Analysis**\n\nGet instant feedback on code quality and security")
            with feat_col3:
                st.info("**📚 Smart Q&A**\n\nAsk questions about your documentation")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # CTA
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("🎯 Get Started", type="primary", use_container_width=True):
                    st.session_state.welcomed = True
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("🔒 Secure • ⚡ Fast • 🎨 Beautiful")
        
        return True
    return False

# Code Generation Tab
def render_generate():
    st.markdown("## 💻 Code Generation Studio")
    st.caption("Describe what you want to build, and let AI write the code")
    st.divider()
    
    # Input Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "What would you like to build?",
            height=180,
            placeholder="e.g., Create a REST API endpoint for user authentication with JWT tokens...",
            help="Be specific about what you want the code to do"
        )
    
    with col2:
        st.markdown("#### ⚙️ Settings")
        lang = st.selectbox("Language", LANGUAGES, help="Target programming language")
        st.markdown("<br>", unsafe_allow_html=True)
        tests = st.toggle("Include Tests", False, help="Generate unit tests")
        docs = st.toggle("Include Docs", True, help="Add documentation")
    
    st.divider()
    
    # Generate Button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        generate_btn = st.button("✨ Generate Code", type="primary", disabled=not prompt, use_container_width=True)
    
    if generate_btn:
        with st.status("🤖 AI is writing code...", expanded=True) as status:
            st.write("🔍 Analyzing requirements...")
            time.sleep(0.5)
            st.write("🧠 Generating code structure...")
            
            try:
                agent = CodeGenerationAgent(st.session_state.selected_model)
                req = CodeGenerationRequest(
                    prompt=prompt,
                    language=Language(lang.lower()),
                    include_tests=tests,
                    include_docs=docs
                )
                res, _ = run_async(agent.generate(req))
                
                st.write("✅ Code generation complete!")
                status.update(label="✅ Successfully generated!", state="complete")
                
                # Results Section
                st.markdown("---")
                st.markdown(f"### 📝 {res.filename}")
                
                # Code display with download
                col_code1, col_code2 = st.columns([4, 1])
                with col_code1:
                    st.code(res.code, language=res.language.value, line_numbers=True)
                with col_code2:
                    st.download_button(
                        "⬇️ Download",
                        res.code,
                        file_name=res.filename,
                        mime="text/plain",
                        use_container_width=True
                    )
                    if st.button("📋 Copy", use_container_width=True):
                        st.success("Copied!")
                
                # Explanation
                if res.explanation:
                    with st.expander("📖 Explanation", expanded=True):
                        st.info(res.explanation)
                
            except Exception as e:
                status.update(label="❌ Generation failed", state="error")
                st.error(f"Error: {str(e)}")
                st.caption("Please try again or modify your prompt")

# Code Analysis Tab
def render_analyze():
    st.markdown("## 🔍 Code Analysis Lab")
    st.caption("Get instant insights on code quality, security, and performance")
    st.divider()
    
    # Input Section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        code = st.text_area(
            "Paste your code here",
            height=300,
            placeholder="# Paste your code here...",
            help="Paste the code you want to analyze"
        )
    
    with col2:
        st.markdown("#### ⚙️ Language")
        lang = st.selectbox("Select", LANGUAGES, key="analyze_lang")
        
        st.markdown("#### 🎯 Focus")
        st.caption("Analysis areas:")
        st.caption("• Security")
        st.caption("• Performance")
        st.caption("• Best Practices")
    
    st.divider()
    
    # Analyze Button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_btn = st.button("🔬 Analyze Code", type="primary", disabled=not code, use_container_width=True)
    
    if analyze_btn:
        with st.status("🔍 Analyzing code...", expanded=True) as status:
            st.write("🔒 Checking security vulnerabilities...")
            time.sleep(0.3)
            st.write("⚡ Analyzing performance...")
            time.sleep(0.3)
            st.write("📊 Evaluating best practices...")
            
            try:
                agent = CodeAnalysisAgent(st.session_state.selected_model)
                req = CodeAnalysisRequest(
                    code=code,
                    language=Language(lang.lower()),
                    focus_areas=["security", "performance", "best_practices"]
                )
                res, _ = run_async(agent.analyze(req))
                
                st.write("✅ Analysis complete!")
                status.update(label="✅ Analysis complete!", state="complete")
                
                # Results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Score Display
                col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
                with col_score2:
                    st.metric(
                        label="Quality Score",
                        value=f"{res.quality_score}/100",
                        delta="Excellent" if res.quality_score >= 80 else "Good" if res.quality_score >= 60 else "Needs Improvement"
                    )
                    st.progress(res.quality_score / 100)
                
                st.divider()
                
                # Summary
                with st.expander("📋 Full Analysis Report", expanded=True):
                    st.markdown(res.summary)
                
                # Additional insights
                col_insights1, col_insights2 = st.columns(2)
                with col_insights1:
                    st.success("**✅ Strengths**")
                    for s in res.strengths:
                        st.caption(f"• {s}")
                
                with col_insights2:
                    st.warning("**💡 Recommendations**")
                    for r in res.recommendations:
                        st.caption(f"• {r}")
                
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(f"Error: {str(e)}")

# Assistant Tab
def render_assistant():
    st.markdown("## 🤖 AI Coding Assistant")
    st.caption("Your personal AI pair programmer - ask anything about code!")
    st.divider()
    
    # Initialize chat
    if "chat_mx" not in st.session_state:
        st.session_state.chat_mx = []
    
    # Welcome message
    if not st.session_state.chat_mx:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""
            👋 Hi! I'm your AI coding assistant. I can help you with:
            
            - Writing code snippets
            - Debugging issues
            - Explaining concepts
            - Suggesting best practices
            - Architecture advice
            
            **What can I help you with today?**
            """)
    
    # Display chat history
    for m in st.session_state.chat_mx:
        avatar = "👤" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about coding...", key="assistant_input"):
        # User message
        st.session_state.chat_mx.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    agent = CodeAssistant(st.session_state.selected_model)
                    res = run_async(agent.chat(prompt))
                    st.markdown(res.message)
                    
                    # Display code snippets
                    for idx, snippet in enumerate(res.code_snippets):
                        st.code(snippet.code, language=snippet.language)
                    
                    st.session_state.chat_mx.append({"role": "assistant", "content": res.message})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# RAG Tab
def render_rag():
    st.markdown("## 📚 Document Q&A")
    st.caption("Ask questions about your uploaded documents")
    st.divider()
    
    # Check for documents
    if not st.session_state.doc_store.list_documents():
        st.info("📁 **No documents uploaded yet**")
        st.markdown("""
        To use this feature:
        1. Upload a PDF document using the sidebar
        2. Click "Process Document"
        3. Start asking questions!
        """)
        return
    
    # Document info
    docs = st.session_state.doc_store.list_documents()
    with st.expander(f"📄 Loaded Documents ({len(docs)})", expanded=False):
        for doc in docs:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.caption(f"📄 {doc['filename']}")
            with col2:
                st.caption(f"{doc['pages']} pages")
            with col3:
                st.caption(f"{doc['chunks']} chunks")
    
    st.divider()
    
    # Initialize chat
    if "rag_mx" not in st.session_state:
        st.session_state.rag_mx = []
    
    # Welcome message
    if not st.session_state.rag_mx:
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(f"""
            I've loaded **{len(docs)} document(s)** with **{st.session_state.vector_store.total_chunks} chunks**.
            
            Ask me anything about your documents!
            """)
    
    # Display chat history
    for m in st.session_state.rag_mx:
        avatar = "👤" if m["role"] == "user" else "📚"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents...", key="rag_input"):
        # User message
        st.session_state.rag_mx.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("Searching documents..."):
                try:
                    agent = RAGAgent(st.session_state.selected_model, st.session_state.vector_store)
                    res, _ = run_async(agent.answer(prompt))
                    
                    st.markdown(res.answer)
                    
                    # Show sources
                    if res.sources:
                        with st.expander(f"📎 Sources ({len(res.sources)})", expanded=False):
                            for idx, source in enumerate(res.sources, 1):
                                st.caption(f"**Source {idx}** (Page {source.page_number})")
                                st.caption(source.content[:200] + "...")
                                st.divider()
                    
                    st.session_state.rag_mx.append({"role": "assistant", "content": res.answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main App
def main():
    render_sidebar()
    
    if render_welcome():
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💻 Code Generator",
        "🔍 Code Analyzer", 
        "🤖 AI Assistant",
        "📚 Doc Q&A"
    ])
    
    with tab1:
        render_generate()
    
    with tab2:
        render_analyze()
    
    with tab3:
        render_assistant()
    
    with tab4:
        render_rag()

if __name__ == "__main__":
    main()
