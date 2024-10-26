import React, { useState, useEffect } from 'react';
import './FlaggedClones.css';

const FlaggedClones = () => {
    const [flaggedClones, setFlaggedClones] = useState([]);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

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
                setFlaggedClones(prevClones => prevClones.filter(clone => clone.flaggedUsername !== flaggedUsername));
            } else {
                console.error('Error deleting clone');
            }
        } catch (error) {
            console.error('Error deleting clone:', error);
        }
    };

    // Handle analysis and open modal
    const handleAnalysis = async (originalUsername, cloneUsername) => {
        try {
            const response = await fetch('http://localhost:5000/analyze-impact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ original_username: originalUsername, clone_username: cloneUsername }),
            });

            if (response.ok) {
                const result = await response.json();
                setAnalysisResult(result);
                setIsModalOpen(true);
            } else {
                console.error('Error performing analysis');
            }
        } catch (error) {
            console.error('Error performing analysis:', error);
        }
    };

    // Close the modal
    const closeModal = () => {
        setIsModalOpen(false);
        setAnalysisResult(null);
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
                            <button
                                className="analysis-button"
                                onClick={() => handleAnalysis(clone.originalUsername, clone.flaggedUsername)}
                            >
                                Impact Analysis
                            </button>
                        </div>
                    </div>
                ))
            ) : (
                <p>No flagged clones found</p>
            )}

            {/* Modal */}
            {isModalOpen && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h4>Impact Analysis Result</h4>
                        {analysisResult && (
                            <>
                                <p><strong>Original Username:</strong> {analysisResult.original_username}</p>
                                <p><strong>Clone Username:</strong> {analysisResult.clone_username}</p>
                                <p><strong>Contribution:</strong> {JSON.stringify(analysisResult.contribution)}</p>
                                <p><strong>Modularity:</strong> {analysisResult.modularity}</p>
                                <p><strong>Influence %:</strong> {analysisResult.influence_percentage}%</p>
                                <p><strong>Clone Community:</strong> {(analysisResult.clone_community || []).join(', ')}</p>
                                <p><strong>Original Community:</strong> {(analysisResult.original_community || []).join(', ')}</p>
                                <p><strong>Intersecting Communities:</strong> {(analysisResult.intersecting_communities || []).join(', ')}</p>
                                <p><strong>Common Neighbors:</strong> {(analysisResult.common_neighbors || []).join(', ')}</p>
                            </>
                        )}
                        <button className="close-button" onClick={closeModal}>Close</button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default FlaggedClones;
