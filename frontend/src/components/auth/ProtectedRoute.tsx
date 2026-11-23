import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAppStore } from "../../stores/useAppStore";

interface ProtectedRouteProps {
  role?: "user" | "admin" | "superadmin";
}

export default function ProtectedRoute({ role = "user" }: ProtectedRouteProps) {
  const { token, user } = useAppStore();
  const location = useLocation();

  if (!token || !user) {
    return <Navigate to="/auth" state={{ from: location.pathname }} replace />;
  }

  if (role === "admin" && user.role !== "admin" && user.role !== "superadmin") {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
}
