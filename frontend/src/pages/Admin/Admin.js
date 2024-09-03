import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Admin.css'; // Import the CSS file

const Admin = () => {
  const navigate = useNavigate();

  const handleViewGraph = () => {
    navigate('/graph');
  };

  const handleFlaggedClones = () => {
    navigate('/flagged-clones');
  };

  return (
    <div className="container">
      <button className="button" onClick={handleViewGraph}>View Graph</button>
      <button className="button" onClick={handleFlaggedClones}>Flagged Clones</button>
    </div>
  );
};

export default Admin;
