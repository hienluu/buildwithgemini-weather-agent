import asyncio
import os
import re
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-bb71fffb09ef")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from simple_agent.agent import root_agent, generate_weather_meme
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

app = FastAPI(title="Interactive Riddle Weather & Meme Agent")

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="simple_agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
AGENT_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "generated_memes")
os.makedirs(AGENT_STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/memes", StaticFiles(directory=AGENT_STATIC_DIR), name="memes")

client_sessions = {}
meme_jobs = {}


class ChatRequest(BaseModel):
    message: str
    client_id: str = "default_user"


async def async_generate_meme(job_id: str, city: str, condition: str, joke_theme: str):
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(
            None,
            generate_weather_meme,
            city,
            condition,
            joke_theme,
        )
        match = re.search(r"filename:\s*(weather_meme_\d+\.png)", res)
        if match:
            filename = match.group(1)
            meme_jobs[job_id] = {"status": "ready", "url": f"../generated_memes/{filename}"}
        else:
            meme_jobs[job_id] = {"status": "failed", "error": res}
    except Exception as exc:
        meme_jobs[job_id] = {"status": "failed", "error": str(exc)}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>LaughCast Riddle App Loading...</h1>")


@app.get("/api/meme-status/{job_id}")
async def get_meme_status(job_id: str):
    if job_id not in meme_jobs:
        return {"status": "not_found"}
    return meme_jobs[job_id]


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    user_id = req.client_id
    if user_id not in client_sessions:
        session = await session_service.create_session(app_name="simple_agent", user_id=user_id)
        client_sessions[user_id] = session.id
    session_id = client_sessions[user_id]

    user_content = Content(role="user", parts=[Part.from_text(text=req.message)])
    replies = []
    try:
        async for event in runner.run_async(session_id=session_id, user_id=user_id, new_message=user_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        replies.append(part.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw_text = "\n\n".join(replies).strip()

    # Parse structured fields
    forecast = ""
    joke_setup = ""
    joke_punchline = ""

    forecast_match = re.search(r"\[FORECAST\]:\s*(.*?)(?=\[JOKE_|$)", raw_text, re.DOTALL | re.IGNORECASE)
    setup_match = re.search(r"\[JOKE_SETUP\]:\s*(.*?)(?=\[JOKE_PUNCHLINE\]|$)", raw_text, re.DOTALL | re.IGNORECASE)
    punchline_match = re.search(r"\[JOKE_PUNCHLINE\]:\s*(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)

    if forecast_match:
        forecast = forecast_match.group(1).strip()
    if setup_match:
        joke_setup = setup_match.group(1).strip()
    if punchline_match:
        joke_punchline = punchline_match.group(1).strip()

    # Fallback if unformatted text
    if not forecast and not joke_setup:
        forecast = raw_text

    meme_job_id = None
    if forecast:
        meme_job_id = str(uuid.uuid4())
        meme_jobs[meme_job_id] = {"status": "pending"}

        city = "the city"
        for word in req.message.replace("?", "").replace("!", "").split():
            if len(word) > 3 and word.istitle():
                city = word
                break

        theme = f"Joke scenario: {joke_setup} Answer: {joke_punchline}" if joke_setup else forecast
        asyncio.create_task(async_generate_meme(meme_job_id, city, "weather condition", theme))

    return {
        "forecast": forecast,
        "joke_setup": joke_setup,
        "joke_punchline": joke_punchline,
        "meme_job_id": meme_job_id,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
