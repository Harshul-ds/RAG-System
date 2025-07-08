"""FastAPI wrapper exposing the agent as a REST endpoint."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import bootstrap_agent

app = FastAPI(title="Agentic RAG API")
try:
    agent = bootstrap_agent()
except Exception as e:
    print(f"[WARN] Failed to initialize full RAG agent: {e}. Falling back to echo mode.")
    agent = None

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    if not query.question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    if agent:
        try:
            result = agent.invoke({"input": query.question})
            return {"answer": result["output"]}
        except Exception as e:
            print(f"[ERROR] Agent invocation failed: {e}")
            raise HTTPException(status_code=500, detail="Agent error")
    # Fallback simple echo if agent missing
    return {"answer": f"(Echo) You asked: {query.question}"}
