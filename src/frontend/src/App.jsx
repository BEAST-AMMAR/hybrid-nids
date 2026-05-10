import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/Login';
import Signup from './components/Signup';
import Monitor from './components/Monitor';
import './index.css';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login setToken={setToken} />} />
        <Route path="/signup" element={<Signup />} />
        <Route 
          path="/monitor" 
          element={token ? <Monitor token={token} setToken={setToken} /> : <Navigate to="/login" />} 
        />
        <Route path="/" element={<Navigate to="/monitor" />} />
      </Routes>
    </Router>
  );
}

export default App;
