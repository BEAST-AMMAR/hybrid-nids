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
from scapy.layers.inet import IP as ScapyIP

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

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

# Resolve model directory relative to this file (works from any CWD)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "saved_models")

# Global instances
detector = HybridDetector(_MODELS_DIR)
extractor = RealTimeFeatureExtractor(
    os.path.join(_MODELS_DIR, "feature_columns.txt"),
    os.path.join(_MODELS_DIR, "standard_scaler.joblib"),
    _MODELS_DIR
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
        
    email = user_info['email']
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        # Create new user via OAuth
        user = models.User(
            email=email,
            full_name=user_info.get('name', ''),
            oauth_provider='google',
            oauth_id=user_info.get('sub'),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    # Redirect to frontend with token
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

from src.backend.traffic_generator import generator_instance
from fastapi import UploadFile, File
from pydantic import BaseModel

class CaptureModeRequest(BaseModel):
    mode: str # 'live', 'virtual_synthetic', 'physical_synthetic'

@app.post("/api/capture/mode")
async def set_capture_mode(request: CaptureModeRequest, current_user: models.User = Depends(auth.get_current_user)):
    mode = request.mode
    if mode == 'live':
        generator_instance.stop()
        return {"status": "success", "message": "Live network sniffing active. Synthetic generator stopped."}
    elif mode == 'virtual_synthetic':
        generator_instance.start(virtual=True, target_queue=packet_queue)
        return {"status": "success", "message": "Virtual synthetic injection started."}
    elif mode == 'physical_synthetic':
        generator_instance.start(virtual=False, target_queue=packet_queue)
        return {"status": "success", "message": "Physical synthetic injection started."}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

import pandas as pd
import io
import time

@app.post("/api/upload_dataset")
async def upload_dataset(
    file: UploadFile = File(...), 
    current_user: models.User = Depends(auth.get_current_user)
):
    if not file.filename.endswith(('.csv', '.txt', '.pcap')):
        raise HTTPException(status_code=400, detail="Only .csv, .txt, or .pcap files are supported")
        
    content = await file.read()
    
    if file.filename.endswith('.pcap'):
        # Pass to scapy reader in a background task
        # For now, just a placeholder or basic implementation
        def process_pcap():
            from scapy.all import rdpcap
            packets = rdpcap(io.BytesIO(content))
            for pkt in packets:
                try:
                    packet_queue.put_nowait(pkt)
                    time.sleep(0.01)
                except queue.Full:
                    pass
        threading.Thread(target=process_pcap, daemon=True).start()
        return {"status": "success", "message": "PCAP processing started. Packets are being pushed to the queue."}
        
    else: # CSV/TXT (NSL-KDD format)
        # Parse NSL-KDD dataset
        try:
            df = pd.read_csv(io.BytesIO(content), header=None)
            
            # Since NSL-KDD has 43 columns usually, we just take the first 41 features 
            # and push them directly to detector bypasses the sniffer feature extraction
            # This is complex because HybridDetector expects scaled features
            # For simplicity, we simulate the network packets based on the rows, or just pass to model
            
            def process_csv():
                # Define column names based on NSL-KDD
                feature_names = [
                    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
                ]
                
                for index, row in df.iterrows():
                    if len(row) >= 41:
                        # Convert row to DataFrame for the extractor
                        features = list(row[:41])
                        
                        # Basic info for the UI
                        protocol_type = str(row[1])
                        service = str(row[2])
                        flag = str(row[3])
                        
                        packet_info = {
                            "src": f"Dataset_Row_{index}",
                            "dst": f"Service_{service}",
                            "proto": protocol_type,
                            "size": int(row[4]) if pd.api.types.is_numeric_dtype(type(row[4])) else 0
                        }
                        
                        # Create row dataframe
                        row_df = pd.DataFrame([features], columns=feature_names)
                        
                        # Encode categorical
                        try:
                            row_df['protocol_type'] = extractor.le_proto.transform(row_df['protocol_type'])
                        except: row_df['protocol_type'] = 0
                            
                        try:
                            row_df['service'] = extractor.le_service.transform(row_df['service'])
                        except: row_df['service'] = 0
                            
                        try:
                            row_df['flag'] = extractor.le_flag.transform(row_df['flag'])
                        except: row_df['flag'] = 0
                        
                        # Scale
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            try:
                                scaled_feat = extractor.scaler.transform(row_df.values)
                                # Get real prediction
                                is_anomaly, score = detector.predict(scaled_feat)
                            except Exception as e:
                                print(f"Scaling/Prediction Error: {e}")
                                continue
                        
                        # Compare with ground truth label if present (index 41)
                        label = str(row[41]) if len(row) > 41 else "unknown"
                        
                        data = {
                            "type": "log",
                            "is_anomaly": bool(is_anomaly),
                            "score": float(score),
                            "packet": packet_info,
                            "explanation": f"Dataset Label: {label}. Predicted Anomaly: {is_anomaly} (Score: {score:.4f})"
                        }
                        
                        if connected_clients:
                            message = json.dumps(data)
                            
                            # We need to broadcast asynchronously from a synchronous thread
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = asyncio.get_event_loop()
                                
                            async def broadcast():
                                tasks = [client.send_text(message) for client in list(connected_clients)]
                                if tasks:
                                    await asyncio.gather(*tasks, return_exceptions=True)
                            
                            asyncio.run_coroutine_threadsafe(broadcast(), loop)
                            
                    time.sleep(0.1) # stream slowly
            
            threading.Thread(target=process_csv, daemon=True).start()
            return {"status": "success", "message": "Dataset processing started."}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    BATCH_SIZE = 32
    while True:
        try:
            if packet_queue.empty():
                await asyncio.sleep(0.01)
                continue
                
            batch = []
            while not packet_queue.empty() and len(batch) < BATCH_SIZE:
                batch.append(packet_queue.get())
                
            # Perform feature extraction in a separate thread
            scaled_feats, valid_infos = await asyncio.to_thread(extractor.packets_to_features, batch)
            
            if scaled_feats is not None and len(scaled_feats) > 0:
                # Perform batch prediction
                is_anomalies, scores = await asyncio.to_thread(detector.predict_batch, scaled_feats)
                
                for i in range(len(valid_infos)):
                    packet_info = valid_infos[i]
                    is_anomaly = is_anomalies[i]
                    score = scores[i]
                    
                    data = {
                        "type": "log",
                        "is_anomaly": is_anomaly,
                        "score": score,
                        "packet": packet_info
                    }
                    
                    if is_anomaly:
                        explanation = await rag_service.explain_anomaly(packet_info, score)
                        data["explanation"] = explanation
                        
                    # Broadcast to all connected clients
                    if connected_clients:
                        message = json.dumps(data)
                        broadcast_tasks = [client.send_text(message) for client in connected_clients]
                        await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            
            for _ in range(len(batch)):
                packet_queue.task_done()
                
            # Always yield to the event loop so other tasks can run
            await asyncio.sleep(0)
            
        except Exception as e:
            import traceback
            print(f"Error in packet processing task: {e}")
            traceback.print_exc()
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
