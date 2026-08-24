import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import AdminPanel from './pages/AdminPanel';
import ProtectedRoute from './components/ProtectedRoute';

const App = () => (
  <Routes>
    <Route path="/login" element={<Login />} />

    <Route
      path="/admin"
      element={
        <ProtectedRoute requireAdmin>
          <AdminPanel />
        </ProtectedRoute>
      }
    />

    {/* Catch-all → send back to /login */}
    <Route path="*" element={<Navigate to="/login" replace />} />
  </Routes>
);

export default App;
