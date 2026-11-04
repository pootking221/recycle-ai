
import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Scan,
  Layers,
  Recycle,
  Menu,
  X,
  ChevronLeft,
} from "lucide-react";

const menus = [
  {
    name: "Detection",
    path: "/",
    icon: Scan,
  },
  {
    name: "Segmentation",
    path: "/segmentation",
    icon: Layers,
  },
];

function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const linkClass = ({ isActive }) =>
    `flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium transition ${
      isActive
        ? "bg-green-700 text-white shadow-sm"
        : "text-gray-600 hover:bg-green-50 hover:text-green-700"
    }`;

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed left-4 top-4 z-[60] flex h-11 w-11 items-center justify-center rounded-xl bg-green-700 text-white shadow-lg md:hidden"
        aria-label="Open menu"
      >
        <Menu size={22} />
      </button>

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 flex flex-col
          border-r border-gray-200 bg-white
          transition-all duration-300
          ${isCollapsed ? "w-20" : "w-64"}
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
          md:sticky md:top-0 md:h-screen md:translate-x-0
        `}
      >
        {/* Logo */}
        <div className="flex h-20 shrink-0 items-center justify-between border-b border-gray-100 px-5">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-green-100 p-2 text-green-700">
              <Recycle size={32} />
            </div>

            {!isCollapsed && (
              <div>
                <h1 className="font-bold text-gray-900">
                  Recycle AI
                </h1>
                <p className="text-xs text-gray-500">
                  Smart Waste System
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => setIsOpen(false)}
            className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 md:hidden"
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-2 overflow-y-auto p-4">
          {!isCollapsed && (
            <p className="mb-3 px-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Main Menu
            </p>
          )}

          {menus.map((menu) => {
            const Icon = menu.icon;

            return (
              <NavLink
                key={menu.path}
                to={menu.path}
                onClick={() => setIsOpen(false)}
                title={isCollapsed ? menu.name : ""}
                className={linkClass}
              >
                <Icon size={20} className="shrink-0" />
                {!isCollapsed && <span>{menu.name}</span>}
              </NavLink>
            );
          })}
        </nav>
      </aside>
    </>
  );
}

export default Sidebar;
