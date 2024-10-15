import React, { useState, useEffect } from 'react';
import './FlaggedClones.css';

const FlaggedClones = () => {
    const [flaggedClones, setFlaggedClones] = useState([]);

    // Fetch flagged clones from the backend on component mount
    useEffect(() => {
        const fetchFlaggedClones = async () => {
            try {
                const response = await fetch('http://localhost:5000/flagged-clones', {
                    method: 'GET',
                });
                const data = await response.json();
                setFlaggedClones(data);
            } catch (error) {
                console.error('Error fetching flagged clones:', error);
            }
        };

        fetchFlaggedClones();
    }, []);

    // Delete a flagged clone
    const handleDeleteClone = async (flaggedUsername) => {
        try {
            const response = await fetch(`http://localhost:5000/flagged-clones/${flaggedUsername}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                // Remove the deleted clone from the UI
                setFlaggedClones(flaggedClones.filter(clone => clone.flaggedUsername !== flaggedUsername));
            } else {
                console.error('Error deleting clone');
            }
        } catch (error) {
            console.error('Error deleting clone:', error);
        }
    };

    return (
        <div className="flagged-clones-container">
            {flaggedClones.length > 0 ? (
                flaggedClones.map((clone, index) => (
                    <div key={index} className="clone-item">
                        <div className="profile-card">
                            <h3>Original Profile</h3>
                            <p><strong>Username:</strong> {clone.originalUsername}</p>
                        </div>

                        <div className="arrow"></div>

                        <div className="clone-card">
                            <h3>Flagged Clone</h3>
                            <p><strong>Username:</strong> {clone.flaggedUsername}</p>
                            <button
                                className="delete-button"
                                onClick={() => handleDeleteClone(clone.flaggedUsername)}
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                ))
            ) : (
                <p>No flagged clones found</p>
            )}
        </div>
    );
};

export default FlaggedClones;
