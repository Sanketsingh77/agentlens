from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os
import re
import tempfile

from debugger import run_debugger
from scenario_generator import run_scenario_generator

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(
    title="AgentLens API",
    description="AI Agent Testing & Evaluation Platform",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# HELPER: JSON extraction
# ─────────────────────────────────────────

def extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


# ─────────────────────────────────────────
# HELPER: analyze one conversation text
# ─────────────────────────────────────────

def analyze_conversation(conversation_text: str):
    prompt = f"""
You are an expert AI conversation quality evaluator.

Analyze this conversation and return ONLY valid JSON, nothing else:

{{
  "tone_score": <number 0-10>,
  "resolution_score": <number 0-10>,
  "hallucination_risk": "<low/medium/high>",
  "missed_opportunities": ["<string>", "<string>"],
  "overall_summary": "<2 sentence summary>",
  "recommendation": "<one actionable improvement>"
}}

Conversation to analyze:
{conversation_text}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI conversation quality evaluator. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )
    raw = response.choices[0].message.content
    cleaned = extract_json_payload(raw)
    return json.loads(cleaned)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AgentLens API",
        "version": "1.0.0",
        "endpoints": [
            "POST /analyze      - Analyze a conversation file",
            "POST /debug        - Debug conversation turn by turn",
            "POST /scenarios    - Generate test scenarios",
            "GET  /reports      - List all saved reports",
            "GET  /reports/{filename} - Get one saved report",
            "GET  /conversations - List all saved conversations",
            "DELETE /reports/{filename} - Delete a saved report",
            "GET  /health       - Health check"
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a .txt conversation file and get a quality report back."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    contents = await file.read()
    conversation_text = contents.decode("utf-8")

    safe_name = os.path.basename(file.filename)

    os.makedirs("conversations", exist_ok=True)
    with open(f"conversations/{safe_name}", "w", encoding="utf-8") as f:
        f.write(conversation_text)

    try:
        report = analyze_conversation(conversation_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_name = safe_name.replace(".txt", "_report.json")
    with open(f"reports/{report_name}", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "filename": safe_name,
        "report": report
    }


@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    """Upload a .txt conversation file and get a turn-by-turn debug report."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    contents = await file.read()
    conversation_text = contents.decode("utf-8")

    safe_name = os.path.basename(file.filename)

    os.makedirs("conversations", exist_ok=True)
    with open(f"conversations/{safe_name}", "w", encoding="utf-8") as f:
        f.write(conversation_text)

    # Write to a temp file so run_debugger can read it
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        dir=tempfile.gettempdir(),
        encoding="utf-8"
    ) as tmp:
        tmp.write(conversation_text)
        tmp_path = tmp.name

    try:
        results = run_debugger(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "filename": safe_name,
        "turns": results
    }


class ScenarioRequest(BaseModel):
    agent_description: str


@app.post("/scenarios")
def scenarios(request: ScenarioRequest):
    """Provide an agent description and get 15 test scenarios back."""
    if not request.agent_description.strip():
        raise HTTPException(status_code=400, detail="agent_description cannot be empty.")

    try:
        result = run_scenario_generator(request.agent_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {str(e)}")

    return {
        "agent_description": request.agent_description,
        "scenarios": result
    }


@app.get("/reports")
def list_reports():
    """List all saved report files."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return {"reports": []}

    files = [f for f in os.listdir(reports_dir) if f.endswith(".json")]
    return {
        "total": len(files),
        "reports": sorted(files)
    }


@app.get("/reports/{filename}")
def get_report(filename: str):
    """Return the content of a specific report file."""
    path = f"reports/{filename}"
    if not os.path.exists(path) or not filename.endswith(".json"):
        raise HTTPException(status_code=404, detail="Report not found.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/conversations")
def list_conversations():
    """List all saved conversation files."""
    folder = "conversations"
    if not os.path.exists(folder):
        return {"conversations": []}
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    return {"total": len(files), "conversations": sorted(files)}


@app.delete("/reports/{filename}")
def delete_report(filename: str):
    """Delete a specific report file."""
    path = f"reports/{filename}"
    if not os.path.exists(path) or not filename.endswith(".json"):
        raise HTTPException(status_code=404, detail="Report not found.")
    os.remove(path)
    return {"deleted": filename}