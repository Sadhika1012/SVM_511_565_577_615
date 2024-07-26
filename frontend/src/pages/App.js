import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Graph from './pages/Graph/Graph';
import Login from './pages/Login/Login';
import Profile from './pages/Profile/Profile';
function App() {  
  return (
    <Router>
      <Routes>
      <Route path="/graph" element={<Graph />} />
      <Route path="/login" element={<Login />} />
      <Route path="/profile" element={<Profile />} />
      </Routes>      
    </Router>
  );
}

export default App;
