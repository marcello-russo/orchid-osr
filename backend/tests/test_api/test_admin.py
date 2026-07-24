import json


class TestAdminDeleteUser:
    """DELETE /auth/users/{username} — richiede ruolo admin/superadmin."""

    def test_superadmin_can_delete_user(self, client, auth_db, superadmin_token):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        client.post(
            "/auth/register",
            json={
                "username": "delete_me",
                "password": "Password1",
                "invite_code": invite["code"],
            },
        )

        response = client.delete(
            "/auth/users/delete_me",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 200
        assert response.json()["deleted"] == "delete_me"

    def test_delete_nonexistent_user(self, client, superadmin_token):
        response = client.delete(
            "/auth/users/nonexistent",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 404

    def test_delete_unauthorized(self, client):
        response = client.delete("/auth/users/someone")
        assert response.status_code == 401

    def test_user_cannot_delete_other(self, client, auth_db, user_token, superadmin_token):
        auth_data = json.loads(auth_db.read_text())
        invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

        client.post(
            "/auth/register",
            json={
                "username": "target_user",
                "password": "Password1",
                "invite_code": invite["code"],
            },
        )

        response = client.delete(
            "/auth/users/target_user",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 403


class TestAdminListUsers:
    """GET /auth/invites — richiede ruolo admin/superadmin, restituisce users e invites."""

    def test_list_users_as_superadmin(self, client, superadmin_token):
        response = client.get(
            "/auth/invites",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert isinstance(data["users"], list)

    def test_list_users_as_user_denied(self, client, user_token):
        response = client.get(
            "/auth/invites",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 403

    def test_list_users_unauthorized(self, client):
        response = client.get("/auth/invites")
        assert response.status_code == 401
