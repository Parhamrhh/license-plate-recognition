import React from 'react';
import { Navigate } from 'react-router-dom';

// Only allow through if a token exists. If requireAdmin is true, also check role.
const isAuthenticated = () => !!sessionStorage.getItem('token');
const isAdmin = () => sessionStorage.getItem('role') === 'admin';

const ProtectedRoute = ({ children, requireAdmin }) => {
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }
  if (requireAdmin && !isAdmin()) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

export default ProtectedRoute;
