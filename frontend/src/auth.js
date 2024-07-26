// auth.js
import { useState, useEffect } from 'react';
import axios from 'axios';

export const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    axios.get('/api/check-auth')
      .then(response => setIsAuthenticated(response.data.authenticated))
      .catch(() => setIsAuthenticated(false));
  }, []);

  return { isAuthenticated };
};
