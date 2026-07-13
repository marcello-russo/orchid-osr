import { useState, useRef, useCallback, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import { useAppStore } from "@/stores/useAppStore";
import { login, register } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { AlertCircle, CheckCircle2, Eye, EyeOff, Loader2, User, Key } from "lucide-react";

// Costruisce schemi con messaggi i18n — saranno ricreati al mount
function createSchemas(t: (key: string) => string) {
  const loginSchema = z.object({
    username: z.string().min(1, t("auth.errors.usernameRequired")),
    password: z.string().min(1, t("auth.errors.passwordRequired")),
  });

  const registerSchema = z
    .object({
      inviteCode: z.string().min(1, t("auth.errors.inviteRequired")),
      username: z.string().min(3, t("auth.errors.usernameLength")).max(32, t("auth.errors.usernameLength")),
      password: z.string().min(8, t("auth.errors.passwordLength")).max(128, t("auth.errors.passwordLength")),
      confirmPassword: z.string(),
    })
    .refine((data) => data.password === data.confirmPassword, {
      message: t("auth.errors.passwordsDontMatch"),
      path: ["confirmPassword"],
    });

  return { loginSchema, registerSchema };
}

type FieldErrors = Record<string, string>;

function getFieldErrors(error: z.ZodError | null): FieldErrors {
  if (!error) return {};
  const result: FieldErrors = {};
  for (const issue of error.issues) {
    const field = issue.path.join(".");
    if (!result[field]) result[field] = issue.message;
  }
  return result;
}

const PASSWORD_REQUIREMENTS = [
  { key: "min8", test: (p: string) => p.length >= 8 },
  { key: "hasUpper", test: (p: string) => /[A-Z]/.test(p) },
  { key: "hasLower", test: (p: string) => /[a-z]/.test(p) },
  { key: "hasNumber", test: (p: string) => /[0-9]/.test(p) },
];

function getPasswordStrength(password: string): { level: number; label: string; color: string } {
  const passed = PASSWORD_REQUIREMENTS.filter((r) => r.test(password)).length;
  if (password.length === 0) return { level: 0, label: "", color: "bg-muted" };
  if (password.length < 8) return { level: 1, label: "auth.passwordStrength.weak", color: "bg-red-500" };
  if (passed <= 2) return { level: 2, label: "auth.passwordStrength.weak", color: "bg-red-500" };
  if (passed <= 3) return { level: 3, label: "auth.passwordStrength.good", color: "bg-yellow-500" };
  return { level: 4, label: "auth.passwordStrength.strong", color: "bg-green-500" };
}

export default function AuthPage() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [inviteCode, setInviteCode] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [zodError, setZodError] = useState<z.ZodError | null>(null);
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const isSubmitting = useRef(false);
  const errorTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const firstErrorRef = useRef<HTMLInputElement>(null);
  const { setAuth } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: string })?.from ?? "/";
  const schemas = useMemo(() => createSchemas(t), [t]);
  const fieldErrors = getFieldErrors(zodError);
  const strength = getPasswordStrength(password);
  const firstErrorField = useMemo(() => {
    const keys = Object.keys(fieldErrors);
    return keys.length > 0 ? keys[0] : null;
  }, [fieldErrors]);

  // Auto-dismiss error dopo 10s
  const setError = useCallback((msg: string) => {
    setApiError(msg);
    if (errorTimer.current) clearTimeout(errorTimer.current);
    errorTimer.current = setTimeout(() => setApiError(""), 10000);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setZodError(null);
    setApiError("");

    if (isSubmitting.current) return;
    isSubmitting.current = true;

    const data =
      mode === "login"
        ? { username, password }
        : { inviteCode, username, password, confirmPassword };

    const schema = mode === "login" ? schemas.loginSchema : schemas.registerSchema;
    const result = schema.safeParse(data);
    if (!result.success) {
      setZodError(result.error);
      isSubmitting.current = false;
      return;
    }

    setLoading(true);
    try {
      const res =
        mode === "login"
          ? await login(username, password)
          : await register(inviteCode, username, password);
      setAuth(res.token, res.user);
      setSuccess(true);
      setTimeout(() => navigate(from, { replace: true }), 300);
    } catch (err: unknown) {
      if (err instanceof Error) {
        const msg = err.message;
        const statusMatch = msg.match(/^API (\d+):/);
        const status = statusMatch ? parseInt(statusMatch[1]) : 0;
        if (status === 429) {
          setError(t("auth.errors.rateLimited"));
        } else if (status === 401) {
          setError(t("auth.errors.invalidCredentials"));
        } else if (status === 409) {
          setError(t("auth.errors.usernameTaken"));
        } else if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
          setError(t("auth.errors.networkError"));
        } else {
          setError(msg);
        }
      } else {
        setError(t("common.error"));
      }
    } finally {
      setLoading(false);
      isSubmitting.current = false;
    }
  };

  const switchMode = () => {
    setMode(mode === "login" ? "register" : "login");
    setZodError(null);
    setApiError("");
  };

  const renderField = (opts: {
    id: string;
    label: string;
    value: string;
    onChange: (v: string) => void;
    type?: string;
    placeholder?: string;
    icon: React.ReactNode;
    required?: boolean;
    rightElement?: React.ReactNode;
  }) => {
    const errorMsg = fieldErrors[opts.id];
    return (
      <div className="space-y-2">
        <Label htmlFor={opts.id}>{opts.label}</Label>
        <div className="relative">
          <div className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground">
            {opts.icon}
          </div>
          <Input
            id={opts.id}
            ref={(el) => {
              if (opts.id === firstErrorField && el) {
                firstErrorRef.current = el;
                el.focus();
              }
            }}
            type={opts.type || "text"}
            value={opts.value}
            onChange={(e) => opts.onChange(e.target.value)}
            className={`pl-10 ${opts.rightElement ? "pr-10" : ""} ${errorMsg ? "border-destructive" : ""}`}
            placeholder={opts.placeholder || opts.label}
            required={opts.required}
            aria-invalid={errorMsg ? "true" : undefined}
            aria-describedby={errorMsg ? `${opts.id}-error` : undefined}
          />
          {opts.rightElement && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              {opts.rightElement}
            </div>
          )}
        </div>
        {errorMsg && (
          <p id={`${opts.id}-error`} className="text-xs text-destructive" role="alert">
            {errorMsg}
          </p>
        )}
      </div>
    );
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
              type="button"
              onClick={switchMode}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "login"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t("auth.loginButton")}
            </button>
            <button
              type="button"
              onClick={switchMode}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "register"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t("auth.registerButton")}
            </button>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            {mode === "register" &&
              renderField({
                id: "inviteCode",
                label: t("auth.inviteCode"),
                value: inviteCode,
                onChange: (v) => setInviteCode(v.toUpperCase()),
                placeholder: t("auth.inviteCode"),
                icon: <Key className="h-4 w-4" />,
                required: true,
              })}

            {renderField({
              id: "username",
              label: t("auth.username"),
              value: username,
              onChange: setUsername,
              icon: <User className="h-4 w-4" />,
              required: true,
            })}

            {renderField({
              id: "password",
              label: t("auth.password"),
              value: password,
              onChange: setPassword,
              type: showPassword ? "text" : "password",
              icon: <Key className="h-4 w-4" />,
              required: true,
              rightElement: (
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="text-muted-foreground hover:text-foreground"
                  tabIndex={-1}
                  aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              ),
            })}

            {mode === "register" && password && (
              <div className="space-y-2">
                <div className="flex gap-1">
                  {[1, 2, 3, 4].map((level) => (
                    <div
                      key={level}
                      className={`h-1 flex-1 rounded-full transition-colors ${
                        level <= strength.level ? strength.color : "bg-muted"
                      }`}
                    />
                  ))}
                </div>
                {strength.label && (
                  <p className="text-xs text-muted-foreground">{t(strength.label)}</p>
                )}
                <div className="grid grid-cols-2 gap-1 text-xs text-muted-foreground">
                  {PASSWORD_REQUIREMENTS.map((req) => {
                    const ok = req.test(password);
                    return (
                      <span key={req.key} className={ok ? "text-green-600" : ""}>
                        {ok ? "✓" : "○"} {req.key}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {mode === "register" &&
              renderField({
                id: "confirmPassword",
                label: t("auth.confirmPassword"),
                value: confirmPassword,
                onChange: setConfirmPassword,
                type: "password",
                icon: <Key className="h-4 w-4" />,
                required: true,
              })}

            {/* API error banner */}
            {apiError && (
              <div role="alert" className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-md">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{apiError}</span>
              </div>
            )}

            {/* Success feedback */}
            {success && (
              <div role="status" className="flex items-center gap-2 text-sm text-green-600 bg-green-50 dark:bg-green-950 p-3 rounded-md">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>{t("common.success")}</span>
              </div>
            )}

            <Button type="submit" className="w-full" disabled={loading || success}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {mode === "login" ? t("auth.loginButton") : t("auth.registerButton")}
            </Button>

            <p className="text-center text-sm text-muted-foreground">
              {mode === "login" ? (
                <>
                  {t("auth.noAccount")}{" "}
                  <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">
                    {t("auth.registerButton")}
                  </button>
                </>
              ) : (
                <>
                  {t("auth.hasAccount")}{" "}
                  <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">
                    {t("auth.loginButton")}
                  </button>
                </>
              )}
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
