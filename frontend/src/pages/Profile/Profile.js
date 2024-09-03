import React,{ useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Profile.css'; // Import the CSS file

const Profile = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { user } = location.state || {}; // Access user data passed from login
    const [clones, setClones] = useState([]); // State to store clones
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
                // For example, you might want to set state to render the results
            } else {
                console.error('Find clones failed:', data.message);
            }
        } catch (error) {
            console.error('Find clones failed', error);
        }};
    
    return (
        <div className="profile-container">
            <div className="profile-header">
                <div className="profile-picture">
                    {/* Add a profile picture here */}
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
            {clones.length > 0 && (
                <div className="clones-list">
                    <h2>Clone Results</h2>
                    <ul>
                        {clones.map((clone, index) => (
                            <li key={index}>
                                <div>
                                    <h3>{clone.profile.name}</h3>
                                    <p><strong>Username:</strong> {clone.profile.username}</p>
                                    <p><strong>Position:</strong> {clone.profile.position}</p>
                                    <p><strong>City:</strong> {clone.profile.city}</p>
                                    <p><strong>About:</strong> {clone.profile.about}</p>
                                    <p><strong>Education:</strong> {clone.profile.education}</p>
                                    <p><strong>Score:</strong> {clone.score.toFixed(2)}</p>
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
