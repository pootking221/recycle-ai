
import { HashRouter, Routes, Route } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";


import Detection from "./pages/Detection";
import Segmentation from "./pages/Segmentation";


function App() {
  return (
    <HashRouter>
      <div className="min-h-screen bg-gray-50 md:flex">

        <Sidebar />

        <div className="min-w-0 flex-1">
          <Navbar />

          <main className="p-4 pt-16 sm:p-6 md:p-8 md:pt-8">
            <Routes>
              <Route path="/" element={<Detection />} />
              <Route path="/segmentation" element={<Segmentation />} />
            </Routes>
          </main>
        </div>

      </div>
    </HashRouter>
  );
}

export default App;
