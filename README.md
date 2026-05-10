# Hybrid ML-based Network Intrusion Detection System (NIDS)

A real-time Network Intrusion Detection System utilizing a Hybrid architecture (Autoencoder + Isolation Forest) and RAG-based anomaly explanations.

## Features
- **Hybrid AI Detection**: Uses Autoencoder reconstruction error and Isolation Forest statistical outliers for robust detection.
- **Real-time Monitoring**: Asynchronous packet sniffing and processing.
- **RAG Explanations**: Automated analysis of detected anomalies based on security knowledge bases.
- **Modern UI**: React-based dashboard with real-time logs and live alerts.
- **Secure Auth**: JWT-based authentication and Google OAuth integration.

## Architecture
- **Frontend**: React (Vite), Tailwind CSS, Lucide Icons, Axios.
- **Backend**: FastAPI, SQLAlchemy (SQLite), Scapy (Packet Sniffing), Scikit-learn, TensorFlow.
- **Model**: Trained on 41 features from the NSL-KDD dataset.

## Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js & npm
- Administrator/Sudo privileges (for packet sniffing)

### Backend Setup
1. Navigate to `src/backend`.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn sqlalchemy scapy pandas numpy scikit-learn tensorflow authlib itsdangerous python-jose[cryptography] passlib[bcrypt]
   ```
3. Set environment variables (create a `.env` file based on `.env.example`):
   ```bash
   JWT_SECRET_KEY=your_secret_key
   ```
4. Run the server:
   ```bash
   # Sniffing requires sudo
   sudo python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. Navigate to `src/frontend`.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Feature Mapping (41 Features)
The system extracts intrinsic and window-based features from live traffic, including:
- `protocol_type`, `service`, `flag`
- `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`
- `count`, `srv_count`, `serror_rate`, `srv_serror_rate` (derived from a 2-second rolling window)

## Security
- All sensitive configurations are managed via environment variables.
- Packet processing is decoupled from sniffing to prevent blocking high-frequency traffic.
