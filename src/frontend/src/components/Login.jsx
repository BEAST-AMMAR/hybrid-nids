import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, Link, useLocation } from 'react-router-dom';

const Login = ({ setToken }) => {
    const location = useLocation();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        // Check for token in URL (from OAuth redirect)
        const params = new URLSearchParams(location.search);
        const token = params.get('token');
        if (token) {
            setToken(token);
            localStorage.setItem('token', token);
            navigate('/monitor');
        }
    }, [location, setToken, navigate]);

    const handleGoogleLogin = () => {
        window.location.href = 'http://localhost:8000/login/google';
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        const params = new URLSearchParams();
        params.append('username', email);
        params.append('password', password);

        try {
            const response = await axios.post('http://localhost:8000/token', params);
            setToken(response.data.access_token);
            localStorage.setItem('token', response.data.access_token);
            navigate('/monitor');
        } catch (err) {
            setError('Invalid email or password');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-900">
            <div className="bg-gray-800 p-8 rounded-lg shadow-xl w-96">
                <h2 className="text-3xl font-bold text-white mb-6 text-center">NIDS Login</h2>
                {error && <p className="text-red-500 mb-4 text-center">{error}</p>}
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label className="block text-gray-400 mb-2">Email</label>
                        <input
                            type="email"
                            className="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block text-gray-400 mb-2">Password</label>
                        <input
                            type="password"
                            className="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded transition duration-200"
                    >
                        Login
                    </button>
                </form>
                <p className="text-gray-400 mt-4 text-center">
                    Don't have an account? <Link to="/signup" className="text-blue-500 hover:underline">Sign up</Link>
                </p>
                <div className="mt-6 flex flex-col space-y-2">
                    <button
                        onClick={handleGoogleLogin}
                        className="bg-white text-gray-700 py-2 rounded flex items-center justify-center font-semibold hover:bg-gray-100 transition"
                    >
                        Sign in with Google
                    </button>
                    <button className="bg-gray-700 text-white py-2 rounded flex items-center justify-center font-semibold border border-gray-600">
                        Sign in with GitHub
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Login;
