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
            // Call the clone detection API
            const cloneResponse = await fetch('http://localhost:5000/find-clones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(user),
                credentials: 'include',
            });

            const cloneData = await cloneResponse.json();

            if (!cloneResponse.ok) {
                console.error('Find clones failed:', cloneData.message);
                return;
            }

            let avatarComparisonResults = [];
            if (user.avatar && user.avatar.trim() !== "") {
                const avatarResponse = await fetch('http://localhost:5000/compare-avatars', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ current_profile: { ...user, avatar_base64: user.avatar } }),
                    credentials: 'include',
                });
        
                const avatarData = await avatarResponse.json();
                if (avatarResponse.ok) {
                    avatarComparisonResults = avatarData.results;
                } else {
                    console.error('Avatar comparison failed:', avatarData.error);
                }
            }
        
            // Combine clone and avatar comparison results
            const combinedResults = cloneData.result.map(clone => {
                const avatarResult = avatarComparisonResults.find(
                    result => result.username === clone.profile.username
                );
                
                const deepfaceScore = avatarResult ? avatarResult.deepface_score.toFixed(2) : 'N/A';
                const ssimScore = avatarResult ? avatarResult.ssim_score.toFixed(2) : 'N/A';
        
                return {
                    ...clone,
                    deepface_score: deepfaceScore,
                    ssim_score: ssimScore,
                };
            });
        
            setClones(combinedResults);
        } catch (error) {
            console.error('Find clones failed', error);
        }
    };
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
    const filteredClones = clones.filter(clone => clone.profile.username !== user.username && clone.score > 0.6);
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
            
            {filteredClones.length > 0 && (
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
                                    <p><strong>Profile Similarity Score:</strong> {clone.score.toFixed(2)}</p>
                                    <p><strong>Profile Picture Similarity Score (DeepFace):</strong> {clone.deepface_score}</p>
                                    <p><strong>Profile Picture Similarity Score (SSIM):</strong> {clone.ssim_score}</p>
                                    <button 
                                        className="flag-button" 
                                        onClick={() => handleFlagClone(clone)}
                                        disabled={flaggedClones.includes(clone.profile.username)}
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
