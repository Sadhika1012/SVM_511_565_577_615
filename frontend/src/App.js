import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Admin from './pages/Admin/Admin';
import Login from './pages/Login/Login';
import Profile from './pages/Profile/Profile';
import Graph from './pages/Graph/Graph';
import FlaggedClones from './pages/Admin/FlaggedClones';
function App() {  
  return (
    <Router>
      <Routes>
      <Route path="/admin" element={<Admin />} />
      <Route path="/login" element={<Login />} />
      <Route path="/profile" element={<Profile />} />
      <Route path="/graph" element={<Graph />} />
      <Route path="/flagged-clones" element={<FlaggedClones />}/>
      </Routes>      
    </Router>
  );
}

export default App;
