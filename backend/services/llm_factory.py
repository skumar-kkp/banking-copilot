from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from backend.core.config import settings

def get_llm(temperature=0.0):
    if settings.LLM_PROVIDER.lower() == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model="claude-3-haiku-20240307", # Faster/cheaper model for general use or specify sonnet
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY
        )
    elif settings.LLM_PROVIDER.lower() == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            # model="gemini-2.5-pro", # Switching to 2.5-pro as older versions are removed
            model="gemini-2.5-flash", # Switching to 2.5-pro as older versions are removed
            temperature=temperature,
            google_api_key=settings.GEMINI_API_KEY
        )
    elif settings.LLM_PROVIDER.lower() == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        return ChatGroq(
            model="llama-3.1-8b-instant", # Example Groq model
            temperature=temperature,
            api_key=settings.GROQ_API_KEY
        )
    else:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY
        )
