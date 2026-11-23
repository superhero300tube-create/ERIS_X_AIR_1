import os
import json
import asyncio
import platform
from datetime import datetime, timezone
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="ERIS-X + Air to Talk Unified OS", version="5.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UnifiedRequest(BaseModel):
    command: str = Field(min_length=1, max_length=12000)
    model_router: str = "groq"
    model_mode: str = "cyber_pink"
    agent_count: int = Field(default=128, ge=1, le=128)

class RAGMemoryStore:
    """Simple process-memory history. It resets when the service restarts."""
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def store(self, topic: str, findings: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "findings": findings,
        }
        self.records.append(entry)
        if len(self.records) > 100:
            self.records.pop(0)
        return entry

memory = RAGMemoryStore()

class PCDesktopAgent:
    def __init__(self):
        self.os_info = platform.platform()

    def diagnostics(self):
        return {
            "os": self.os_info,
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": "SERVER_ENVIRONMENT_ONLY",
            "note": "Render par yeh Render server ka environment hai, aapke personal PC ka nahi.",
        }

    def log_task(self, task_name: str):
        # On Render this local filesystem may be temporary and is not the user's PC.
        workspace = os.path.join(APP_DIR, "workspace")
        os.makedirs(workspace, exist_ok=True)
        log_path = os.path.join(workspace, "task_audit.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().astimezone().isoformat()}] {task_name}\n")
        return log_path

pc_agent = PCDesktopAgent()

async def ask_groq(user_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY Render Environment mein set nahi hai. API key add karke service redeploy karein.",
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ERIS-X Air to Talk, a helpful conversational assistant. "
                    "Reply in the language used by the user; if they ask for Marathi, use Marathi. "
                    "Be clear and honest. Do not claim that simulated agents performed real work, "
                    "do not claim access to the user's PC, camera, CCTV, files, or live web unless "
                    "a connected tool actually provides that access."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.6,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0)) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        if response.status_code == 401:
            raise HTTPException(status_code=502, detail="Groq ne API key reject ki. Render mein GROQ_API_KEY check karein.")
        if response.status_code == 429:
            raise HTTPException(status_code=503, detail="Groq rate limit ya quota limit aa gayi. Thodi der baad try karein ya Groq account usage check karein.")
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", {}).get("message", response.text)
            except Exception:
                message = response.text
            raise HTTPException(status_code=502, detail=f"Groq API error ({response.status_code}): {message[:500]}")
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Groq se khaali response mila.")
        return answer
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq response mein timeout hua. Dobara try karein.")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Groq connection error: {str(exc)[:300]}")

@app.get("/")
def serve_portal():
    html_path = os.path.join(APP_DIR, "index.html")
    if not os.path.isfile(html_path):
        raise HTTPException(status_code=404, detail="index.html file nahi mili.")
    return FileResponse(html_path)

@app.get("/health")
def health_check():
    return {
        "status": "operational",
        "system": "ERIS-X + Air to Talk",
        "ai_provider_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
    }

@app.get("/api/pc-diagnostics")
def get_pc_diagnostics():
    return pc_agent.diagnostics()

@app.get("/api/memory")
def get_memory():
    return {"count": len(memory.records), "stored_entries": memory.records, "storage": "process_memory_only"}

@app.post("/api/unified-execute")
async def unified_execute(req: UnifiedRequest):
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=422, detail="Command khaali nahi ho sakta.")

    # Real answer comes from Groq. The 128-node label below is explicitly a simulation,
    # not 128 separately hosted or independently reasoning AI agents.
    answer = await ask_groq(command)
    pc_agent.log_task(command)
    memory.store(command, answer)
    simulated_count = req.agent_count
    summary = (
        f"Groq AI response generated successfully. Agent visualization: "
        f"{simulated_count} virtual nodes (simulation only; not {simulated_count} independent AI agents)."
    )
    return {
        "status": "success",
        "router": "groq",
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        "model_mode": req.model_mode,
        "speech_response": answer,
        "expression": "talking_smile",
        "swarm_summary": summary,
        "memory_count": len(memory.records),
        "sample_nodes": [],
        "agent_simulation": True,
    }

@app.websocket("/ws/air-talk-live")
async def air_talk_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)
            user_text = str(packet.get("text", "")).strip()
            if not user_text:
                await websocket.send_text(json.dumps({"type": "error", "text": "Message khaali hai."}))
                continue
            try:
                answer = await ask_groq(user_text)
                await websocket.send_text(json.dumps({"type": "ai_response", "text": answer, "expression": "speaking"}, ensure_ascii=False))
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"type": "error", "text": str(exc.detail)}, ensure_ascii=False))
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
