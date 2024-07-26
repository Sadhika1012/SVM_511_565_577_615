import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Network } from 'vis-network/standalone/umd/vis-network.min.js';

function Graph() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const networkContainerRef = useRef(null);

  useEffect(() => {
    axios.get('http://localhost:5000/graph')
      .then(response => {
        const nodes = response.data.nodes.map((node, id) => ({ id: node.id, label: node.label }));
        const edges = response.data.edges.map((edge, id) => ({ from: edge.from , to: edge.to, id:edge.id}));
        setGraphData({ nodes, edges });
      })
      .catch(error => {
        console.error('There was an error fetching the graph data!', error);
      });
  }, []);

  useEffect(() => {
    if (networkContainerRef.current && graphData.nodes.length > 0) {
      const network = new Network(networkContainerRef.current, graphData, {});
    }
  }, [graphData]);

  return (
    <>
      <h1>Graph Visualization</h1>
      <div ref={networkContainerRef} style={{ height: '400px' }}></div>
    </>
  );
}

export default Graph;