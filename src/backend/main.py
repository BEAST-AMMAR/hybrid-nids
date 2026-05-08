from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth
from datetime import timedelta
from typing import List
import asyncio
import threading
import json
import queue

from . import models, schemas, auth, database
from .database import engine, get_db
from .sniffer import RealTimeFeatureExtractor, start_sniffing
from .detector import HybridDetector
from .rag_service import RAGExplanationService

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hybrid NIDS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=auth.SECRET_KEY)

# OAuth Setup
oauth = OAuth()
import os
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name='github',
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url='https://github.com/login/oauth/access_token',
        access_token_params=None,
        authorize_url='https://github.com/login/oauth/authorize',
        authorize_params=None,
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'user:email'},
    )

# Global instances
detector = HybridDetector("src/backend/saved_models")
extractor = RealTimeFeatureExtractor(
    "src/backend/saved_models/feature_columns.txt",
    "src/backend/saved_models/standard_scaler.joblib",
    "src/backend/saved_models"
)
rag_service = RAGExplanationService()

# WebSocket clients
connected_clients = set()
# Background processing queue
packet_queue = queue.Queue()

@app.post("/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password, full_name=user.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/login/google")
async def login_google(request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/google")
async def auth_google(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=401, detail="Google auth failed")

    user_info = token.get('userinfo')
    if not user_info:
        raise HTTPException(status_code=401, detail="No user info from Google")

    return await handle_oauth_user(user_info['email'], user_info.get('name', ''), 'google', user_info.get('sub'), db)

@app.get("/login/github")
async def login_github(request: Request):
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    redirect_uri = request.url_for('auth_github')
    return await oauth.github.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/github")
async def auth_github(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.github.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=401, detail="GitHub auth failed")

    resp = await oauth.github.get('user', token=token)
    user_info = resp.json()

    email = user_info.get('email')
    if not email:
        # If email is not public, get it from emails endpoint
        emails_resp = await oauth.github.get('user/emails', token=token)
        emails = emails_resp.json()
        email = next((e['email'] for e in emails if e['primary']), emails[0]['email'])

    return await handle_oauth_user(email, user_info.get('name') or user_info.get('login'), 'github', str(user_info.get('id')), db)

async def handle_oauth_user(email: str, name: str, provider: str, provider_id: str, db: Session):
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        user = models.User(
            email=email,
            full_name=name,
            oauth_provider=provider,
            oauth_id=provider_id,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"http://localhost:5173/login?token={access_token}")

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket client connected")
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

def packet_callback(packet):
    # This runs in the high-frequency sniffing thread.
    # Put the packet into a queue and return immediately.
    try:
        packet_queue.put_nowait(packet)
    except queue.Full:
        pass # Drop packet if queue is full to prevent blocking

async def process_packets():
    """Background task to process packets from the queue without blocking the sniffer."""
    while True:
        try:
            # Get packet from queue
            if not packet_queue.empty():
                packet = packet_queue.get()

                # Perform feature extraction and detection
                scaled_feat = extractor.packet_to_features(packet)
                if scaled_feat is not None:
                    is_anomaly, score = detector.predict(scaled_feat)

                    packet_info = {
                        "src": packet[0][1].src if packet.haslayer("IP") else "Unknown",
                        "dst": packet[0][1].dst if packet.haslayer("IP") else "Unknown",
                        "proto": packet[0][1].proto if packet.haslayer("IP") else "Unknown",
                        "size": len(packet)
                    }

                    data = {
                        "type": "log",
                        "is_anomaly": is_anomaly,
                        "score": score,
                        "packet": packet_info
                    }

                    if is_anomaly:
                        # RAG explanation is async, await it here in the non-blocking task
                        explanation = await rag_service.explain_anomaly(packet_info, score)
                        data["explanation"] = explanation

                    # Broadcast to all connected clients
                    if connected_clients:
                        message = json.dumps(data)
                        # Create list of tasks to broadcast
                        broadcast_tasks = [client.send_text(message) for client in connected_clients]
                        await asyncio.gather(*broadcast_tasks, return_exceptions=True)

                packet_queue.task_done()
            else:
                await asyncio.sleep(0.01) # Yield if queue is empty
        except Exception as e:
            print(f"Error in packet processing task: {e}")
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    # Start sniffing in a background thread
    sniffer_thread = threading.Thread(target=start_sniffing, args=(packet_callback,), daemon=True)
    sniffer_thread.start()

    # Start the async processing task
    asyncio.create_task(process_packets())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
