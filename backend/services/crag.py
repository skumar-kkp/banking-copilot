import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from backend.services.llm_factory import get_llm
from backend.services.guardrails import check_compliance_and_guardrails
from backend.core.config import settings
from backend.data.lookup import StructuredLookup

# Initialize structured lookup
_lookup = StructuredLookup()

# Load FAISS index lazily
_vectorstore = None
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        persist_dir = "data/vectorstore"
        if os.path.exists(persist_dir):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            _vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore

class GraderOutput(BaseModel):
    score: float = Field(description="Relevance score between 0.0 and 1.0")
    is_relevant: bool = Field(description="Whether the chunk is relevant to the query")

def grade_chunk(query: str, chunk_content: str) -> float:
    llm = get_llm(temperature=0.0).with_structured_output(GraderOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance grader. Grade the retrieved chunk on its relevance to the user query.
Assess two dimensions:
1. Topical relevance: Does it address the query? (Note: "penalty", "charges", and "fees" are often used interchangeably in banking).
2. Specificity: Is it precise enough to ground a safe answer in banking?
Return a score between 0.0 and 1.0. If the chunk is highly relevant and specific, return > 0.6. If vague or irrelevant, return < 0.6.
Also set is_relevant to true if score >= 0.6."""),
        ("user", "Query: {query}\n\nChunk: {chunk}")
    ])
    try:
        chain = prompt | llm
        result = chain.invoke({"query": query, "chunk": chunk_content})
        
        # Handle both Pydantic objects and dictionaries
        if hasattr(result, 'score'):
            return result.score
        elif isinstance(result, dict) and 'score' in result:
            return result['score']
        return 0.0
    except Exception as e:
        print(f"Grader error: {e}")
        return 0.0

def rewrite_query(query: str) -> str:
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query rewriting assistant for a Banking AI. Rewrite the original query to be broader or use different terminology to improve retrieval. Output ONLY the rewritten query string, nothing else."),
        ("user", "Original query: {query}")
    ])
    try:
        chain = prompt | llm
        result = chain.invoke({"query": query})
        return result.content.strip()
    except:
        return query

def generate_answer(query: str, context: str, is_low_confidence: bool = False) -> str:
    llm = get_llm(temperature=0.0)
    system_msg = """You are a Banking AI Copilot. Answer the user query using ONLY the provided context.
If you don't know the answer based on the context, say so.
Cite specific clauses, document names, or page numbers when possible."""
    
    if is_low_confidence:
        system_msg += "\nNOTE: The context provided is from a web search fallback. Explicitly state 'Low confidence — based on web search, not verified policy document.' at the beginning of your response."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Context: {context}\n\nQuery: {query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"context": context, "query": query})
    return result.content

async def run_crag_pipeline(original_query: str, session_id: str) -> Dict[str, Any]:
    trace = {"steps": []}
    current_query = original_query
    max_retries = 2
    iteration = 0
    
    vectorstore = get_vectorstore()
    
    retrieved_docs = []
    final_docs = []
    is_low_confidence = False
    
    # Structured Data Lookup (CSV)
    structured_context = ""
    query_lower = original_query.lower()
    if any(keyword in query_lower for keyword in ["rate", "fee", "charge", "interest", "eligibility", "maximum loan"]):
        trace["steps"].append({"action": "Structured Data Lookup Triggered"})
        if "eligibility" in query_lower or "maximum loan" in query_lower:
            structured_context = _lookup.lookup_loan_eligibility(original_query)
        else:
            structured_context = _lookup.lookup_interest_rate(original_query)
        
        if "No matching" not in structured_context and "No specific" not in structured_context:
            trace["steps"].append({"action": "Structured Lookup Result", "found": True})
            # If we found structured data, we add it to the final docs as a virtual document
            final_docs.append(Document(page_content=structured_context, metadata={"source": "Official Rate/Eligibility Schedule"}))
        else:
            trace["steps"].append({"action": "Structured Lookup Result", "found": False})

    if vectorstore is not None and not final_docs:
        while iteration <= max_retries:
            trace["steps"].append({"action": f"Retrieve iteration {iteration}", "query": current_query})
            
            # Retrieve
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(current_query)
            
            # Grade
            relevant_docs = []
            total_score = 0
            chunk_previews = []
            for doc in retrieved_docs:
                score = grade_chunk(original_query, doc.page_content)
                if score >= settings.GRADING_THRESHOLD:
                    relevant_docs.append(doc)
                total_score += score
                chunk_previews.append(doc.page_content[:100].replace("\n", " ") + "...")
                
            avg_score = total_score / max(len(retrieved_docs), 1)
            trace["steps"].append({
                "action": "Grade", 
                "avg_score": avg_score, 
                "passed": avg_score >= settings.GRADING_THRESHOLD,
                "retrieved_snippets": chunk_previews
            })
            
            if avg_score >= settings.GRADING_THRESHOLD and relevant_docs:
                final_docs.extend(relevant_docs)
                break
            else:
                iteration += 1
                if iteration <= max_retries:
                    current_query = rewrite_query(current_query)
                    trace["steps"].append({"action": "Rewrite Query", "new_query": current_query})

    # Web search fallback
    if not final_docs:
        trace["steps"].append({"action": "Web Search Fallback Triggered"})
        search = DuckDuckGoSearchRun()
        try:
            search_result = search.invoke(original_query)
            final_docs = [Document(page_content=search_result, metadata={"source": "Web Search"})]
            is_low_confidence = True
        except Exception as e:
            search_result = "No results found."
            
    # Combine context
    context = "\n\n".join([doc.page_content for doc in final_docs])
    source_names = [doc.metadata.get("source_file", doc.metadata.get("source", "Unknown")) for doc in final_docs]
    
    # Generate
    trace["steps"].append({"action": "Generate Answer"})
    draft_answer = generate_answer(original_query, context, is_low_confidence)
    
    # Compliance Guardrails
    trace["steps"].append({"action": "Compliance Check"})
    compliance_result = check_compliance_and_guardrails(original_query, draft_answer)
    trace["steps"][-1]["is_compliant"] = compliance_result.get("is_compliant", False)
    trace["steps"][-1]["reason"] = compliance_result.get("reason", "No reason provided")
    
    return {
        "answer": compliance_result.get("final_answer", draft_answer),
        "confidence_level": "low" if is_low_confidence else "high",
        "compliance_status": "passed" if compliance_result.get("is_compliant", False) else "flagged",
        "sources": list(set(source_names)),
        "trace": trace
    }
