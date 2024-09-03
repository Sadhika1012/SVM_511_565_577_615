import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Admin from './pages/Admin/Admin';
import Login from './pages/Login/Login';
import Profile from './pages/Profile/Profile';
import Graph from './pages/Graph/Graph';
function App() {  
  return (
    <Router>
      <Routes>
      <Route path="/admin" element={<Admin />} />
      <Route path="/login" element={<Login />} />
      <Route path="/profile" element={<Profile />} />
      <Route path="/graph" element={<Graph />} />
      </Routes>      
    </Router>
  );
}

export default App;
