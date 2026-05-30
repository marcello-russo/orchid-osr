import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAppStore } from "@/stores/useAppStore";
import { login, register } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { AlertCircle, Eye, EyeOff, Loader2, User, Key } from "lucide-react";

export default function AuthPage() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [inviteCode, setInviteCode] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setAuth } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: string })?.from || "/";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (mode === "register" && password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);
    try {
      const result = mode === "login"
        ? await login(username, password)
        : await register(inviteCode, username, password);
      setAuth(result.token, result.user);
      navigate(from, { replace: true });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setMode(mode === "login" ? "register" : "login");
    setError("");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center pb-2">
          <div className="mb-4">
            <span className="text-primary font-bold text-xl">Orch</span>
            <span className="text-foreground font-bold text-xl">ID</span>
            <span className="text-muted-foreground font-bold text-xs ml-0.5">NCD</span>
          </div>
          <div className="flex rounded-lg bg-muted p-1">
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "login" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Accedi
            </button>
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "register" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Registrati
            </button>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="invite">Codice invito</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="invite"
                    value={inviteCode}
                    onChange={(e) => setInviteCode(e.target.value.toUpperCase())}
                    className="pl-10 font-mono"
                    placeholder="INV-..."
                    required
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-10"
                  placeholder="username"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 pr-10"
                  placeholder="password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="confirm">Conferma password</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="confirm"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="pl-10"
                    placeholder="password"
                    required
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {mode === "login" ? "Accedi" : "Registrati"}
            </Button>

            <p className="text-center text-sm text-muted-foreground">
              {mode === "login" ? (
                <>Non hai un account? <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">Registrati</button></>
              ) : (
                <>Hai già un account? <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">Accedi</button></>
              )}
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
