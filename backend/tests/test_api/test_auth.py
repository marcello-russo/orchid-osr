import json


class TestRegister:
    """POST /auth/register — usa fixture auth_db per database temporaneo."""

    def test_register_success(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        response = client.post(
            "/auth/register",
            json={
                "username": "newuser",
                "password": "Password1",
                "invite_code": invite["code"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["user"]["username"] == "newuser"
        assert data["user"]["role"] == "user"

    def test_register_duplicate_username(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invites = [i for i in auth_data["invites"] if not i["used"] and i["role"] == "user"]
        invite1, invite2 = invites[:2]

        client.post(
            "/auth/register",
            json={
                "username": "dupuser",
                "password": "Password1",
                "invite_code": invite1["code"],
            },
        )
        response = client.post(
            "/auth/register",
            json={
                "username": "dupuser",
                "password": "Password1",
                "invite_code": invite2["code"],
            },
        )
        assert response.status_code == 409
        assert "already taken" in response.text.lower()

    def test_register_weak_password(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        response = client.post(
            "/auth/register",
            json={
                "username": "weakuser",
                "password": "weakpassword",
                "invite_code": invite["code"],
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


class TestLogin:
    """POST /auth/login."""

    def test_login_success(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        client.post(
            "/auth/register",
            json={
                "username": "loginuser",
                "password": "Password1",
                "invite_code": invite["code"],
            },
        )
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

    def test_login_wrong_password(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        client.post(
            "/auth/register",
            json={
                "username": "wrongpwuser",
                "password": "Password1",
                "invite_code": invite["code"],
            },
        )
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

    def test_me_authenticated(self, client, auth_db):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        client.post(
            "/auth/register",
            json={
                "username": "meuser",
                "password": "Password1",
                "invite_code": invite["code"],
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
