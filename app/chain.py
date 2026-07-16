# app/chain.py

from operator import itemgetter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from ingestion.vectorstore import load_vectorstore

# app/chain.py

# 1. Update the Pydantic class to make sources optional with a default value
class ResearchAnswer(BaseModel):
    answer: str = Field(description="The synthesized answer to the user's question")
    sources: list[str] = Field(
        default=[], # <-- FIX: Default to empty list to prevent Pydantic crashes
        description="List of source filenames used to answer (e.g. ['Bandi_Bhupesh_AI Engineer_Resume.pdf'])"
    )

# 2. Update the system prompt to give the LLM clear structured rules
SYSTEM_PROMPT = """You are a research assistant. 

Answer the user's question using ONLY the provided context below. If the context does not contain enough information to answer confidently, say so explicitly.

IMPORTANT: You must return a structured response with:
1. "answer": A string containing the synthesized answer.
2. "sources": A list of strings containing only the exact filenames of the sources you used (e.g., ["Bandi_Bhupesh_AI Engineer_Resume.pdf"]). Do not put the source names inside the answer text itself.

Context:
{context}
"""



prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

def format_docs(docs: list) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# Local chat history store
_session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-sonnet-5", 
        model_kwargs={"temperature": None}
    )
    structured_llm = llm.with_structured_output(ResearchAnswer)

    # 1. Base structured generation chain
    generation_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | structured_llm
    )
    
    # 2. Custom step that takes the Pydantic output, saves the answer text 
    # to the session history, and passes the Pydantic object right through.
    def save_output_to_history(inputs_and_outputs):
        # We need the session_id to know which history to update
        session_id = inputs_and_outputs["session_id"]
        question = inputs_and_outputs["question"]
        output_pydantic = inputs_and_outputs["output"]
        
        # Grab the target session history
        history = get_session_history(session_id)
        
        # Log the raw text exchanges into the chat history manually
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=output_pydantic.answer))
        
        # Return the original structured Pydantic object to main.py
        return output_pydantic

    # 3. Put it all together in a single custom Runnable
    def run_chain_with_history(inputs: dict, config: dict):
        # Resolve configurable session ID
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id", "default")
        
        # Get history to inject into the prompt
        history = get_session_history(session_id)
        
        # Invoke generation with the loaded message history
        output = generation_chain.invoke({
            "question": inputs["question"],
            "history": history.messages
        }, config=config)
        
        # Route to our custom tracker to update memory state before returning
        return save_output_to_history({
            "session_id": session_id,
            "question": inputs["question"],
            "output": output
        })

    return RunnableLambda(run_chain_with_history)


# Keep your entrypoint clean
def build_chain_with_memory():
    return build_chain()