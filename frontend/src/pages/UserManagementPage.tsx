import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { fetchInvites, generateInvites, deleteUser } from "@/lib/api";
import { useAppStore } from "@/stores/useAppStore";
import { useShallow } from "zustand/react/shallow";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Copy, Plus, Users } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

interface Invite {
  code: string;
  role: string;
  used: boolean;
  used_by: string | null;
  created_at: string | null;
}

interface AppUser {
  username: string;
  role: string;
}

export function UserManagementPage() {
  const { t } = useTranslation();
  const [invites, setInvites] = useState<Invite[]>([]);
  const [users, setUsers] = useState<AppUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [count, setCount] = useState(1);
  const [role, setRole] = useState("user");
  const [generating, setGenerating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const currentUser = useAppStore(
    useShallow((s) => ({ username: s.user?.username, role: s.user?.role }))
  );

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchInvites();
      setInvites(data.invites);
      setUsers(data.users);
    } catch (e) {
      console.error("Failed to load invites", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      await generateInvites(count, role);
      await load();
    } catch (e) {
      console.error("Failed to generate invites", e);
    } finally {
      setGenerating(false);
    }
  };

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  async function confirmDelete() {
    if (!deleteTarget) return;
    try {
      await deleteUser(deleteTarget);
      setDeleteTarget(null);
      const data = await fetchInvites();
      setUsers(data.users);
    } catch {
      setDeleteTarget(null);
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary shrink-0">
          <Users className="size-5" />
        </div>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            {t("invites.title")}
          </h1>
          <p className="text-muted-foreground">{t("invites.description")}</p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{t("invites.generate_title")}</CardTitle>
          <CardDescription>
            {t("invites.generate_description")}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-end gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium">
                {t("invites.count")}
              </label>
              <Input
                type="number"
                min={1}
                max={10}
                value={count}
                onChange={(e) =>
                  setCount(Math.max(1, Math.min(10, Number(e.target.value))))
                }
                className="w-20"
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">
                {t("invites.role")}
              </label>
              <Tabs value={role} onValueChange={setRole}>
                <TabsList>
                  <TabsTrigger value="user">{t("admin_users.role_user")}</TabsTrigger>
                  <TabsTrigger value="admin">{t("admin_users.role_admin")}</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <Button onClick={handleGenerate} disabled={generating}>
              <Plus className="mr-2 size-4" />
              {generating ? t("common.loading") : t("invites.generate")}
            </Button>

          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t("invites.list_title")}</CardTitle>
          <CardDescription>
            {t("invites.list_description", {
              total: invites.length,
              available: invites.filter((i) => !i.used).length,
              used: invites.filter((i) => i.used).length,
              users: users.length,
            })}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">
              {t("common.loading")}
            </div>
          ) : invites.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              {t("invites.empty")}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("invites.code")}</TableHead>
                  <TableHead>{t("invites.role")}</TableHead>
                  <TableHead>{t("invites.status")}</TableHead>
                  <TableHead>{t("invites.created_at")}</TableHead>
                  <TableHead>{t("invites.used_by")}</TableHead>
                  <TableHead className="w-10"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {invites.map((inv) => (
                  <TableRow key={inv.code}>
                    <TableCell className="font-mono text-xs">
                      {inv.code}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          inv.role === "admin" ? "default" : "secondary"
                        }
                      >
                        {inv.role}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={inv.used ? "outline" : "secondary"}
                        className={
                          inv.used
                            ? "text-muted-foreground"
                            : "bg-green-100 text-green-800 hover:bg-green-100"
                        }
                      >
                        {inv.used
                          ? t("invites.status_used")
                          : t("invites.status_available")}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {inv.created_at
                        ? new Date(inv.created_at).toLocaleDateString()
                        : "—"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {inv.used_by || "—"}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => copyCode(inv.code)}
                        title={t("common.copy")}
                      >
                        <Copy className="size-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Users className="size-5 text-muted-foreground" />
            <CardTitle>{t("invites.users_title")}</CardTitle>
          </div>
          <CardDescription>
            {t("invites.users_description", { count: users.length })}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("invites.username")}</TableHead>
                <TableHead>{t("invites.role")}</TableHead>
                <TableHead className="text-right">{t("admin_users.actions")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.map((u) => (
                <TableRow key={u.username}>
                  <TableCell className="font-medium">{u.username}</TableCell>
                  <TableCell>
                    <Badge
                      variant={u.role === "admin" ? "default" : "secondary"}
                    >
                      {u.role}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    {(currentUser.role === "superadmin" || (currentUser.role === "admin" && u.role === "user")) && (
                      <Button variant="destructive" size="sm" onClick={() => setDeleteTarget(u.username)}>
                        {t("admin_users.delete")}
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("admin_users.delete_confirm_title")}</DialogTitle>
            <DialogDescription>
              {t("admin_users.delete_confirm_body", { username: deleteTarget })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              {t("common.cancel")}
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              {t("admin_users.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
