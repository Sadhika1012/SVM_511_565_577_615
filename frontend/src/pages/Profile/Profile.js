import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Profile.css'; // Import the CSS file

const Profile = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { user } = location.state || {}; // Access user data passed from login

    // Handle user not present in state (e.g., direct access to profile route)
    if (!user) {
        // Optionally, redirect to login if user data is not found
        navigate('/login');
        return <p>Loading...</p>;
    }

    return (
        <div className="profile-container">
            <div className="profile-header">
                <div className="profile-picture">
                    {/* Add a profile picture here */}
                </div>
                <div className="profile-info">
                    <h2>{user.name}</h2>
                    <p className="position">{user.position}</p>
                    <p className="city">{user.city}</p>
                </div>
            </div>
            <div className="profile-content">
                {/* Additional content can go here */}
            </div>
            <button className="logout-button" onClick={() => navigate('/login')}>Logout</button>
        </div>
    );
};

export default Profile;
