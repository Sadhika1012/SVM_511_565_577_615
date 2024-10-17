import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Profile.css'; // Import the CSS file
import def_img from '../../assets/linkedin_img.jpg';

const Profile = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { user } = location.state || {}; // Access user data passed from login
    const [clones, setClones] = useState([]); // State to store clones
    const [flaggedClones, setFlaggedClones] = useState([]); // To track flagged clones

    // Handle user not present in state (e.g., direct access to profile route)
    if (!user) {
        // Optionally, redirect to login if user data is not found
        navigate('/login');
        return <p>Loading...</p>;
    }

    const handleLogout = async () => {
        try {
            const response = await fetch('http://localhost:5000/logout', { 
                method: 'POST', 
                credentials: 'include' 
            });
    
            const data = await response.json();
    
            // Redirect to login regardless of the response
            if (response.ok) {
                navigate('/login');
            } else {
                console.error('Logout failed:', data.message);
            }
        } catch (error) {
            console.error('Logout failed', error);
        }
    };

    const handleFindClones = async () => {
        try {
            const response = await fetch('http://localhost:5000/find-clones', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(user),
                credentials: 'include', // Include credentials if needed
            });

            const data = await response.json();

            if (response.ok) {
                console.log('Clones found:', data.result);
                // Handle displaying clones here
                setClones(data.result); // Update state with the results
            } else {
                console.error('Find clones failed:', data.message);
            }
        } catch (error) {
            console.error('Find clones failed', error);
        }
    };

    // Filter out the user's own profile from the clones list
    const filteredClones = clones.filter(
        clone => clone.profile.username !== user.username && clone.score > 0.7
    );

    const handleFlagClone = async (clone) => {
        try {
            const response = await fetch('http://localhost:5000/flagged-clones', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    originalUsername: user.username,
                    flaggedUsername: clone.profile.username,
                }),
            });
    
            if (response.ok) {
                const data = await response.json();
                console.log('Clone flagged successfully:', data);
                // Mark the clone as flagged
                setFlaggedClones([...flaggedClones, clone.profile.username]);
            } else {
                const errorData = await response.json();
                console.error('Error flagging clone:', errorData.message);
            }
        } catch (error) {
            console.error('Error flagging clone:', error);
        }
    };

    return (
        <div className="profile-container">
            <div className="profile-header">
                <div className="profile-picture">
                    {user.avatar ? (
                        <img src={`data:image/jpg;base64,${user.avatar}`} alt="Profile Avatar" />
                    ) : (
                        <img src={def_img} alt="Default Avatar" />
                    )}
                </div>

                <div className="profile-info">
                    <h2>{user.username}</h2>
                    <p>{user.name}</p>
                    <p className="position">{user.position}</p>
                    <p className="city">{user.city}</p>
                    <p className="about">{user.about}</p>
                    <p className="education">{user.education}</p>
                </div>
            </div>
            <div className="profile-content">
                {/* Additional content can go here */}
            </div>
            <button className="logout-button" onClick={handleLogout}>Logout</button>
            <button className="find-clones-button" onClick={handleFindClones}>Find Clones</button>
            
            {/* Conditionally render clone results only after the button is clicked */}
            {clones.length > 0 && (
                <div className="clones-list">
                    <h2>Clone Results</h2>
                    <ul>
                        {filteredClones.map((clone, index) => (
                            <li key={index} className="clone-item">
                                <div className="clone-info">
                                    <h3>{clone.profile.name}</h3>
                                    <p><strong>Username:</strong> {clone.profile.username}</p>
                                    <p><strong>Position:</strong> {clone.profile.position}</p>
                                    <p><strong>City:</strong> {clone.profile.city}</p>
                                    <p><strong>About:</strong> {clone.profile.about}</p>
                                    <p><strong>Education:</strong> {clone.profile.education}</p>
                                    <p><strong>Score:</strong> {clone.score.toFixed(2)}</p>
                                    <button 
                                        className="flag-button" 
                                        onClick={() => handleFlagClone(clone)}
                                        disabled={flaggedClones.includes(clone.profile.username)} // Disable after flagging
                                    >
                                        {flaggedClones.includes(clone.profile.username) ? 'Flagged' : 'Flag Clone'}
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Profile;
