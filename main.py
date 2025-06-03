from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from typing import Literal
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis or DB in production)
session_memory = {}

class ChatRequest(BaseModel):
    role: Literal["user", "system"]
    message: str
    session_id: str = "default"  # optional field for user/session tracking

@app.get("/")
def read_root():
    return {"API": "Memory Chatbot"}

@app.post("/chat")
def chat(request: ChatRequest):
    # Load or create memory for session
    if request.session_id not in session_memory:
        session_memory[request.session_id] = ConversationBufferMemory()
    
    memory = session_memory[request.session_id]

    # Use ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Build a conversation chain with memory
    conversation = ConversationChain(
        llm=model,
        memory=memory,
        verbose=True  # Optional: log memory events
    )

    # System message can be added at beginning manually, not per message
    if request.role == "system":
        # Set initial system context manually
        memory.chat_memory.messages.insert(0, 
            {"type": "system", "content": request.message})
        return {"response": "System message set."}

    # Process user message
    result = conversation.predict(input=request.message)
    return {"response": result}
