import React, { useEffect, useState } from 'react';
import { Shield, ShieldAlert, Activity, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Monitor = ({ token, setToken }) => {
    const [logs, setLogs] = useState([]);
    const [status, setStatus] = useState('Disconnected');
    const navigate = useNavigate();

    useEffect(() => {
        if (!token) {
            navigate('/login');
            return;
        }

        const socket = new WebSocket('ws://localhost:8000/ws/monitor');

        socket.onopen = () => {
            setStatus('Monitoring Live Traffic');
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setLogs((prev) => [data, ...prev].slice(0, 50));
        };

        socket.onclose = () => {
            setStatus('Disconnected');
        };

        return () => socket.close();
    }, [token, navigate]);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setToken(null);
        navigate('/login');
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
                        <Activity className={status === 'Disconnected' ? "text-red-500" : "text-green-500"} />
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
                            <div className="text-gray-500 text-center py-20 bg-gray-800 rounded-lg">
                                Waiting for network traffic...
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
                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                        <h3 className="text-lg font-bold mb-4">Security Overview</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Total Packets</span>
                                <span className="font-mono">{logs.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Anomalies</span>
                                <span className="font-mono text-red-500">{logs.filter(l => l.is_anomaly).length}</span>
                            </div>
                            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-blue-500 transition-all duration-500"
                                    style={{ width: logs.length > 0 ? `${(logs.filter(l => !l.is_anomaly).length / logs.length) * 100}%` : '0%' }}
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
