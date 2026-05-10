import React, { useEffect, useState } from 'react';
import { Shield, ShieldAlert, Activity, LogOut, Play, Square, Clock, Server } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Monitor = ({ token, setToken }) => {
    const [logs, setLogs] = useState([]);
    const [status, setStatus] = useState('Disconnected');
    
    // New state for accurate counting and modes
    const [totalPackets, setTotalPackets] = useState(0);
    const [totalAnomalies, setTotalAnomalies] = useState(0);
    const [captureMode, setCaptureMode] = useState('continuous');
    const [trafficSource, setTrafficSource] = useState('live'); // 'live', 'virtual_synthetic', 'physical_synthetic', 'dataset'
    const [selectedFile, setSelectedFile] = useState(null);
    const [isCapturing, setIsCapturing] = useState(false);
    const [timeLeft, setTimeLeft] = useState(null);
    
    const navigate = useNavigate();

    // Timer logic
    useEffect(() => {
        let timer;
        if (isCapturing && timeLeft !== null && timeLeft > 0) {
            timer = setInterval(() => {
                setTimeLeft(prev => prev - 1);
            }, 1000);
        } else if (isCapturing && timeLeft === 0) {
            setIsCapturing(false);
            setStatus('Capture Complete');
        }
        return () => clearInterval(timer);
    }, [isCapturing, timeLeft]);

    // WebSocket logic
    useEffect(() => {
        if (!token) {
            navigate('/login');
            return;
        }

        if (!isCapturing) {
            setStatus(timeLeft === 0 ? 'Capture Complete' : 'Ready / Stopped');
            return;
        }

        const socket = new WebSocket('ws://localhost:8000/ws/monitor');

        socket.onopen = () => {
            setStatus('Monitoring Live Traffic');
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Increment global counters
            setTotalPackets(prev => prev + 1);
            if (data.is_anomaly) {
                setTotalAnomalies(prev => prev + 1);
            }
            
            // Keep UI logs sliced to prevent memory overflow
            setLogs((prev) => [data, ...prev].slice(0, 100));
        };

        socket.onclose = () => {
            setStatus('Disconnected');
        };

        return () => socket.close();
    }, [token, navigate, isCapturing]);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setToken(null);
        navigate('/login');
    };

    const handleStart = async () => {
        try {
            if (trafficSource !== 'dataset') {
                await axios.post('http://localhost:8000/api/capture/mode', { mode: trafficSource }, {
                    headers: { Authorization: `Bearer ${token}` }
                });
            } else {
                if (!selectedFile) {
                    alert("Please select a dataset file (.csv, .txt, or .pcap) first.");
                    return;
                }
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                await axios.post('http://localhost:8000/api/upload_dataset', formData, {
                    headers: { 
                        Authorization: `Bearer ${token}`,
                        'Content-Type': 'multipart/form-data'
                    }
                });
            }
        } catch (error) {
            console.error("Failed to set capture mode or upload dataset", error);
            alert("Failed to start capture: " + (error.response?.data?.detail || error.message));
            return;
        }

        setLogs([]);
        setTotalPackets(0);
        setTotalAnomalies(0);
        if (captureMode === '1m') setTimeLeft(60);
        else if (captureMode === '5m') setTimeLeft(300);
        else setTimeLeft(null);
        setIsCapturing(true);
    };

    const handleStop = async () => {
        setIsCapturing(false);
        if (timeLeft > 0) setTimeLeft(null);
        
        try {
            // Revert back to live mode on stop if we were generating synthetic traffic
            await axios.post('http://localhost:8000/api/capture/mode', { mode: 'live' }, {
                headers: { Authorization: `Bearer ${token}` }
            });
        } catch (error) {
            console.error("Failed to reset capture mode", error);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <header className="flex justify-between items-center mb-8 bg-gray-800 p-4 rounded-lg shadow-md">
                <div className="flex items-center space-x-3">
                    <Shield className="text-blue-500 w-8 h-8" />
                    <h1 className="text-2xl font-bold">Hybrid NIDS Monitor</h1>
                </div>
                <div className="flex items-center space-x-6">
                    <div className="flex items-center space-x-2">
                        <Activity className={status.includes('Stopped') || status.includes('Complete') || status === 'Disconnected' ? "text-red-500" : "text-green-500"} />
                        <span className="font-medium">{status}</span>
                    </div>
                    <button 
                        onClick={handleLogout}
                        className="flex items-center space-x-1 text-gray-400 hover:text-white transition"
                    >
                        <LogOut size={20} />
                        <span>Logout</span>
                    </button>
                </div>
            </header>

            <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-4">
                    <h2 className="text-xl font-semibold mb-4 flex items-center">
                        Real-time Traffic Logs
                    </h2>
                    <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
                        {logs.length === 0 && (
                            <div className="text-gray-500 text-center py-20 bg-gray-800 rounded-lg border border-dashed border-gray-600">
                                {isCapturing ? "Waiting for network traffic..." : "Click 'Start Capture' to begin monitoring."}
                            </div>
                        )}
                        {logs.map((log, index) => (
                            <div key={index} className={`p-4 rounded-lg border ${log.is_anomaly ? 'bg-red-900/20 border-red-500' : 'bg-gray-800 border-gray-700'}`}>
                                <div className="flex justify-between items-start">
                                    <div className="flex items-center space-x-3">
                                        {log.is_anomaly ? <ShieldAlert className="text-red-500" /> : <Shield className="text-green-500" />}
                                        <div>
                                            <p className="font-mono text-sm">
                                                <span className="text-blue-400">{log.packet.src}</span> 
                                                <span className="text-gray-500 mx-2">→</span> 
                                                <span className="text-purple-400">{log.packet.dst}</span>
                                            </p>
                                            <p className="text-xs text-gray-400 mt-1">
                                                Proto: {log.packet.proto} | Size: {log.packet.size} bytes | Score: {log.score.toFixed(4)}
                                            </p>
                                        </div>
                                    </div>
                                    {log.is_anomaly && (
                                        <span className="bg-red-600 text-xs font-bold px-2 py-1 rounded uppercase animate-pulse">
                                            Anomaly
                                        </span>
                                    )}
                                </div>
                                {log.explanation && (
                                    <div className="mt-3 p-3 bg-black/40 rounded border-l-4 border-yellow-500 text-sm italic text-gray-300">
                                        <p className="font-bold text-yellow-500 mb-1">RAG Analysis:</p>
                                        {log.explanation}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="space-y-6">
                    {/* Capture Controls */}
                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                        <h3 className="text-lg font-bold mb-4 flex items-center">
                            <Clock className="w-5 h-5 mr-2 text-blue-400" /> Capture Controls
                        </h3>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Traffic Source</label>
                                <select 
                                    value={trafficSource} 
                                    onChange={(e) => setTrafficSource(e.target.value)}
                                    disabled={isCapturing}
                                    className="w-full bg-gray-700 text-white rounded p-2 border border-gray-600 focus:outline-none focus:border-blue-500 mb-3"
                                >
                                    <option value="live">Live Network (Interface)</option>
                                    <option value="virtual_synthetic">Virtual Synthetic Generator</option>
                                    <option value="physical_synthetic">Physical Synthetic Injection</option>
                                    <option value="dataset">Dataset Upload (.csv, .txt, .pcap)</option>
                                </select>
                            </div>
                            
                            {trafficSource === 'dataset' && (
                                <div className="mb-3">
                                    <label className="block text-sm text-gray-400 mb-1">Select File</label>
                                    <input 
                                        type="file" 
                                        accept=".csv,.txt,.pcap"
                                        onChange={(e) => setSelectedFile(e.target.files[0])}
                                        disabled={isCapturing}
                                        className="w-full bg-gray-700 text-white rounded p-2 border border-gray-600 focus:outline-none focus:border-blue-500"
                                    />
                                </div>
                            )}

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Capture Mode</label>
                                <select 
                                    value={captureMode} 
                                    onChange={(e) => setCaptureMode(e.target.value)}
                                    disabled={isCapturing}
                                    className="w-full bg-gray-700 text-white rounded p-2 border border-gray-600 focus:outline-none focus:border-blue-500"
                                >
                                    <option value="continuous">Continuous (Default)</option>
                                    <option value="1m">1 Minute Session</option>
                                    <option value="5m">5 Minute Session</option>
                                </select>
                            </div>
                            
                            {timeLeft !== null && (
                                <div className="text-center p-3 bg-gray-900 rounded border border-gray-600 font-mono text-xl text-blue-400 shadow-inner">
                                    {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')} remaining
                                </div>
                            )}

                            <div className="flex space-x-3 pt-2">
                                {!isCapturing ? (
                                    <button 
                                        onClick={handleStart}
                                        className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 rounded flex items-center justify-center transition"
                                    >
                                        <Play className="w-5 h-5 mr-2" /> Start Capture
                                    </button>
                                ) : (
                                    <button 
                                        onClick={handleStop}
                                        className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 rounded flex items-center justify-center transition"
                                    >
                                        <Square className="w-5 h-5 mr-2" /> Stop Capture
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                        <h3 className="text-lg font-bold mb-4">Security Overview</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Total Packets</span>
                                <span className="font-mono">{totalPackets}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Anomalies</span>
                                <span className="font-mono text-red-500">{totalAnomalies}</span>
                            </div>
                            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div 
                                    className="h-full bg-blue-500 transition-all duration-500" 
                                    style={{ width: totalPackets > 0 ? `${((totalPackets - totalAnomalies) / totalPackets) * 100}%` : '0%' }}
                                ></div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-blue-900/20 p-6 rounded-lg border border-blue-500/30">
                        <h3 className="text-blue-400 font-bold mb-2">System Info</h3>
                        <p className="text-sm text-gray-300">
                            The engine is utilizing a Hybrid Autoencoder + Isolation Forest architecture for statistical and reconstruction-based anomaly detection.
                        </p>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Monitor;
