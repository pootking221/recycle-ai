
import { Bell, Recycle, UserCircle } from "lucide-react";

function Navbar() {
  return (
    <header className="sticky top-0 z-20 flex h-20 items-center justify-between border-b border-gray-200 bg-white px-5 md:px-8">

      {/* Left */}
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-green-100 p-2.5 text-green-700">
          <Recycle size={32} />
        </div>

        <div>
          <h2 className="text-lg font-bold text-gray-900">
            Recycle AI
          </h2>
          <p className="text-xs text-gray-500">
            Waste Detection System
          </p>
        </div>
      </div>

      {/* Right */}
      <div className="flex items-center gap-5">
        <div className="hidden text-right sm:block">
          <p className="text-xs text-gray-500">
            Local Processing
          </p>
        </div>
      </div>

    </header>
  );
}

export default Navbar;
