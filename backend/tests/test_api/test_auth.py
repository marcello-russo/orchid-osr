class TestRegister:
    """POST /auth/register — usa fixture auth_db (SQLite temporaneo)."""

    def test_register_success(self, client, auth_db, next_invite):
        response = client.post(
            "/auth/register",
            json={
                "username": "newuser",
                "password": "Password1",
                "invite_code": next_invite(),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["user"]["username"] == "newuser"
        assert data["user"]["role"] == "user"

    def test_register_duplicate_username(self, client, auth_db, next_invite):
        client.post(
            "/auth/register",
            json={
                "username": "dupuser",
                "password": "Password1",
                "invite_code": next_invite(),
            },
        )
        # secondo invito distinto: il primo è già consumato
        invite2 = next_invite()
        response = client.post(
            "/auth/register",
            json={
                "username": "dupuser",
                "password": "Password1",
                "invite_code": invite2,
            },
        )
        assert response.status_code == 409
        assert "already taken" in response.text.lower()

    def test_register_weak_password(self, client, auth_db, next_invite):
        response = client.post(
            "/auth/register",
            json={
                "username": "weakuser",
                "password": "weakpassword",
                "invite_code": next_invite(),
            },
        )
        assert response.status_code == 422

    def test_register_invalid_invite(self, client, auth_db):
        response = client.post(
            "/auth/register",
            json={
                "username": "badinviteuser",
                "password": "Password1",
                "invite_code": "FAKE-INV-CODE",
            },
        )
        assert response.status_code == 400
        assert "Invalid invite" in response.text

    def test_register_malformed_expiry_is_rejected(self, client, auth_db, next_invite):
        """Fail-closed: una `expires_at` illeggibile non rende l'invito eterno."""
        from app.core import auth_store

        code = next_invite()
        with auth_store.transaction() as conn:
            conn.execute("UPDATE invites SET expires_at = 'not-a-date' WHERE code = ?", (code,))

        response = client.post(
            "/auth/register",
            json={"username": "maltime", "password": "Password1", "invite_code": code},
        )
        assert response.status_code == 400
        assert "invite code invalid" in response.text.lower()

    def test_register_single_use_invite_is_atomic(self, auth_db):
        """Due registrazioni concorrenti sullo stesso invito: una sola vince."""
        import threading

        from app.core import auth as auth_module
        from app.core import auth_store

        code = auth_module.generate_invite(role="user", count=1)[0]
        outcomes: list[str] = []
        lock = threading.Lock()

        def attempt(n: int) -> None:
            try:
                auth_module.register_user(code, f"racer{n}", "Password1")
                outcome = "ok"
            except Exception:
                outcome = "rejected"
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=attempt, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert outcomes.count("ok") == 1, outcomes
        assert sum(1 for u in auth_store.list_users() if u["username"].startswith("racer")) == 1


class TestLogin:
    """POST /auth/login."""

    def _register(self, client, invite_code: str, username: str) -> None:
        client.post(
            "/auth/register",
            json={"username": username, "password": "Password1", "invite_code": invite_code},
        )

    def test_login_success(self, client, auth_db, next_invite):
        self._register(client, next_invite(), "loginuser")
        response = client.post(
            "/auth/login",
            json={
                "username": "loginuser",
                "password": "Password1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["user"]["username"] == "loginuser"

    def test_login_wrong_password(self, client, auth_db, next_invite):
        self._register(client, next_invite(), "wrongpwuser")
        response = client.post(
            "/auth/login",
            json={
                "username": "wrongpwuser",
                "password": "WrongPassword1",
            },
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client, auth_db):
        response = client.post(
            "/auth/login",
            json={
                "username": "nonexistent",
                "password": "Password1",
            },
        )
        assert response.status_code == 401


class TestMe:
    """GET /auth/me."""

    def test_me_authenticated(self, client, auth_db, next_invite):
        client.post(
            "/auth/register",
            json={
                "username": "meuser",
                "password": "Password1",
                "invite_code": next_invite(),
            },
        )
        login_resp = client.post(
            "/auth/login",
            json={
                "username": "meuser",
                "password": "Password1",
            },
        )
        token = login_resp.json()["token"]

        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "meuser"

    def test_me_unauthenticated(self, client, auth_db):
        response = client.get("/auth/me")
        assert response.status_code == 401


class TestDefensePass:
    """POST /auth/invites/defense-pass + revoke (solo admin)."""

    def test_create_and_reuse_and_revoke(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        resp = client.post(
            "/auth/invites/defense-pass",
            json={"role": "user", "max_uses": 3, "expires_in_hours": 24, "label": "difesa"},
            headers=h,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["max_uses"] == 3 and body["expires_at"] and body["label"] == "difesa"
        code = body["code"]
        for n in range(3):
            r = client.post(
                "/auth/register",
                json={"username": f"dif{n}", "password": "Password1", "invite_code": code},
            )
            assert r.status_code == 200, r.text
        r = client.post(
            "/auth/register",
            json={"username": "dif3", "password": "Password1", "invite_code": code},
        )
        assert r.status_code == 400

    def test_revoke_kills_pass(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        code = client.post("/auth/invites/defense-pass", json={}, headers=h).json()["code"]
        r = client.post(f"/auth/invites/{code}/revoke", headers=h)
        assert r.status_code == 200
        r = client.post(
            "/auth/register",
            json={"username": "revoked1", "password": "Password1", "invite_code": code},
        )
        assert r.status_code == 400
        assert "revoked" in r.text.lower()
        r = client.post("/auth/invites/NOPE-CODE/revoke", headers=h)
        assert r.status_code == 404

    def test_non_admin_cannot_create(self, client, auth_db, user_token):
        h = {"Authorization": f"Bearer {user_token}"}
        r = client.post("/auth/invites/defense-pass", json={}, headers=h)
        assert r.status_code == 403

    def test_admin_role_rejected(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        r = client.post("/auth/invites/defense-pass", json={"role": "admin"}, headers=h)
        assert r.status_code == 422


class TestInviteExpiry:
    """Un invito non deve vivere per sempre, e il cap non dipende dall'endpoint."""

    def test_generated_invite_expires_by_default(self, client, auth_db, superadmin_token):
        from app.core import auth_store

        h = {"Authorization": f"Bearer {superadmin_token}"}
        resp = client.post("/auth/invites/generate", json={"count": 1}, headers=h)
        assert resp.status_code == 200
        code = resp.json()["codes"][0]
        invite = next(i for i in auth_store.list_invites() if i["code"] == code)
        assert invite["expires_at"] is not None

    def test_defense_pass_expires_by_default(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        body = client.post("/auth/invites/defense-pass", json={}, headers=h).json()
        assert body["expires_at"] is not None

    def test_invite_ttl_cannot_exceed_30_days(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        r = client.post("/auth/invites/generate", json={"count": 1, "expires_in_hours": 721}, headers=h)
        assert r.status_code == 422
        r = client.post("/auth/invites/defense-pass", json={"expires_in_hours": 721}, headers=h)
        assert r.status_code == 422

    def test_invite_ttl_cannot_be_zero(self, client, auth_db, superadmin_token):
        """0 = "eterno" resta possibile solo dal CLI, non da un client HTTP."""
        h = {"Authorization": f"Bearer {superadmin_token}"}
        r = client.post("/auth/invites/generate", json={"count": 1, "expires_in_hours": 0}, headers=h)
        assert r.status_code == 422

    def test_guest_pass_cap_is_seven_days(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        assert client.post("/auth/guest-pass", json={"expires_in_hours": 168}, headers=h).status_code == 200
        r = client.post("/auth/guest-pass", json={"expires_in_hours": 169}, headers=h)
        assert r.status_code == 422


class TestGuestEnter:
    """POST /auth/guest-pass + /auth/guest/enter + revoke (solo admin per creare)."""

    def test_full_flow(self, client, auth_db, superadmin_token):
        h = {"Authorization": f"Bearer {superadmin_token}"}
        resp = client.post("/auth/guest-pass", json={"expires_in_hours": 24, "label": "difesa"}, headers=h)
        assert resp.status_code == 200
        body = resp.json()
        assert body["pass"] and body["expires_at"]
        r = client.post("/auth/guest/enter", json={"token": body["pass"]})
        assert r.status_code == 200
        assert r.json()["user"] == {"username": "ospite", "role": "user"}
        assert "token" in r.json()
        r = client.post(f"/auth/guest/revoke/{body['jti']}", headers=h)
        assert r.status_code == 200
        r = client.post("/auth/guest/enter", json={"token": body["pass"]})
        assert r.status_code == 401

    def test_non_admin_cannot_create(self, client, auth_db, user_token):
        h = {"Authorization": f"Bearer {user_token}"}
        r = client.post("/auth/guest-pass", json={}, headers=h)
        assert r.status_code == 403
