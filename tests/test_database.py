"""
tests/test_database.py — Tests for the SQLite persistence layer.
Uses an in-memory SQLite database so no files are created on disk.
All tests are isolated and can run in any order.
"""
import pytest
import os
import sys
import tempfile

@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Point database module at a fresh temp file for each test."""
    db_path = str(tmp_path / "test_tutor.db")
    monkeypatch.setenv("DB_PATH", db_path)
    if 'server.database' in sys.modules:
        del sys.modules['server.database']
    import server.database as db
    db.init_db()
    return db


class TestUserAuth:

    def test_register_new_user(self, temp_db):
        db = temp_db
        user = db.register_user("alice")
        assert user is not None
        assert user["username"] == "alice"
        assert user["token"].startswith("")  
        assert len(user["token"]) > 10
        assert "id" in user

    def test_register_duplicate_username_returns_none(self, temp_db):
        db = temp_db
        db.register_user("alice")
        result = db.register_user("alice")
        assert result is None

    def test_register_username_is_lowercased(self, temp_db):
        db = temp_db
        user = db.register_user("ALICE")
        assert user["username"] == "alice"

    def test_register_username_is_stripped(self, temp_db):
        db = temp_db
        user = db.register_user("  bob  ")
        assert user["username"] == "bob"

    def test_get_user_by_token_valid(self, temp_db):
        db = temp_db
        user = db.register_user("charlie")
        found = db.get_user_by_token(user["token"])
        assert found is not None
        assert found["username"] == "charlie"
        assert found["id"] == user["id"]

    def test_get_user_by_token_invalid(self, temp_db):
        db = temp_db
        result = db.get_user_by_token("nonexistent_token_xyz")
        assert result is None

    def test_get_user_by_username_valid(self, temp_db):
        db = temp_db
        db.register_user("dave")
        found = db.get_user_by_username("dave")
        assert found is not None
        assert found["username"] == "dave"

    def test_get_user_by_username_not_found(self, temp_db):
        db = temp_db
        result = db.get_user_by_username("nobody")
        assert result is None

    def test_get_user_by_username_case_insensitive(self, temp_db):
        db = temp_db
        db.register_user("eve")
        found = db.get_user_by_username("EVE")
        assert found is not None

    def test_multiple_users_have_unique_tokens(self, temp_db):
        db = temp_db
        u1 = db.register_user("user1")
        u2 = db.register_user("user2")
        u3 = db.register_user("user3")
        tokens = {u1["token"], u2["token"], u3["token"]}
        assert len(tokens) == 3 


class TestProblemSessions:

    def test_save_and_load_code(self, temp_db):
        db = temp_db
        user = db.register_user("frank")
        db.save_code(user["id"], "rec1", "sumList [] = 0\nsumList (x:xs) = x + sumList xs")
        code = db.get_saved_code(user["id"], "rec1")
        assert "sumList" in code

    def test_save_code_overwrites_previous(self, temp_db):
        db = temp_db
        user = db.register_user("grace")
        db.save_code(user["id"], "rec1", "old code")
        db.save_code(user["id"], "rec1", "new code")
        code = db.get_saved_code(user["id"], "rec1")
        assert code == "new code"

    def test_get_saved_code_returns_empty_for_unknown_problem(self, temp_db):
        db = temp_db
        user = db.register_user("henry")
        code = db.get_saved_code(user["id"], "nonexistent")
        assert code == ""

    def test_mark_problem_solved(self, temp_db):
        db = temp_db
        user = db.register_user("iris")
        db.mark_problem_solved(user["id"], "rec1")
        progress = db.get_user_progress(user["id"])
        assert "rec1" in progress["solved"]

    def test_mark_problem_solved_twice_no_duplicate(self, temp_db):
        db = temp_db
        user = db.register_user("jack")
        db.mark_problem_solved(user["id"], "rec1")
        db.mark_problem_solved(user["id"], "rec1")
        progress = db.get_user_progress(user["id"])
        assert progress["solved"].count("rec1") == 1

    def test_get_user_progress_empty(self, temp_db):
        db = temp_db
        user = db.register_user("kate")
        progress = db.get_user_progress(user["id"])
        assert progress["solved"] == []
        assert progress["in_progress"] == {}
        assert progress["total_solved"] == 0

    def test_get_user_progress_tracks_solved_count(self, temp_db):
        db = temp_db
        user = db.register_user("leo")
        db.mark_problem_solved(user["id"], "rec1")
        db.mark_problem_solved(user["id"], "rec2")
        db.mark_problem_solved(user["id"], "pat1")
        progress = db.get_user_progress(user["id"])
        assert progress["total_solved"] == 3

    def test_in_progress_shows_unsolved_with_code(self, temp_db):
        db = temp_db
        user = db.register_user("mia")
        db.save_code(user["id"], "rec1", "partial code")
        progress = db.get_user_progress(user["id"])
        assert "rec1" in progress["in_progress"]
        assert progress["in_progress"]["rec1"] == "partial code"

    def test_solved_not_shown_in_in_progress(self, temp_db):
        db = temp_db
        user = db.register_user("noah")
        db.save_code(user["id"], "rec1", "some code")
        db.mark_problem_solved(user["id"], "rec1")
        progress = db.get_user_progress(user["id"])
        assert "rec1" in progress["solved"]
        assert "rec1" not in progress["in_progress"]

    def test_progress_isolated_between_users(self, temp_db):
        db = temp_db
        u1 = db.register_user("olivia")
        u2 = db.register_user("peter")
        db.mark_problem_solved(u1["id"], "rec1")
        p1 = db.get_user_progress(u1["id"])
        p2 = db.get_user_progress(u2["id"])
        assert "rec1" in p1["solved"]
        assert "rec1" not in p2["solved"]


class TestConversations:

    def test_save_and_retrieve_messages(self, temp_db):
        db = temp_db
        user = db.register_user("quinn")
        db.save_message(user["id"], "rec1", "assistant", "What is the base case?")
        db.save_message(user["id"], "rec1", "user",      "When the list is empty")
        db.save_message(user["id"], "rec1", "assistant", "Right! So what does sumList [] return?")
        convo = db.get_conversation(user["id"], "rec1")
        assert len(convo) == 3
        assert convo[0]["role"] == "assistant"
        assert convo[1]["role"] == "user"
        assert convo[2]["role"] == "assistant"

    def test_conversation_content_is_preserved(self, temp_db):
        db = temp_db
        user = db.register_user("rose")
        db.save_message(user["id"], "lc1", "assistant", "What type does evenSquares return?")
        convo = db.get_conversation(user["id"], "lc1")
        assert convo[0]["content"] == "What type does evenSquares return?"

    def test_conversation_context_separation(self, temp_db):
        db = temp_db
        user = db.register_user("sam")
        db.save_message(user["id"], "rec1", "assistant", "Error question", context="error")
        db.save_message(user["id"], "rec1", "assistant", "Think question", context="think")
        error_convo = db.get_conversation(user["id"], "rec1", context="error")
        think_convo = db.get_conversation(user["id"], "rec1", context="think")
        assert len(error_convo) == 1
        assert len(think_convo) == 1
        assert error_convo[0]["content"] == "Error question"
        assert think_convo[0]["content"] == "Think question"

    def test_conversation_empty_for_new_problem(self, temp_db):
        db = temp_db
        user = db.register_user("tina")
        convo = db.get_conversation(user["id"], "rec1")
        assert convo == []

    def test_conversation_limit(self, temp_db):
        db = temp_db
        user = db.register_user("uma")
        for i in range(25):
            db.save_message(user["id"], "rec1", "user" if i%2 else "assistant", f"Message {i}")
        convo = db.get_conversation(user["id"], "rec1", limit=20)
        assert len(convo) == 20

    def test_clear_conversation(self, temp_db):
        db = temp_db
        user = db.register_user("victor")
        db.save_message(user["id"], "rec1", "assistant", "Question 1")
        db.save_message(user["id"], "rec1", "user",      "Answer 1")
        db.clear_conversation(user["id"], "rec1")
        convo = db.get_conversation(user["id"], "rec1")
        assert convo == []

    def test_clear_conversation_only_clears_specified_context(self, temp_db):
        db = temp_db
        user = db.register_user("wendy")
        db.save_message(user["id"], "rec1", "assistant", "Error msg", context="error")
        db.save_message(user["id"], "rec1", "assistant", "Think msg", context="think")
        db.clear_conversation(user["id"], "rec1", context="error")
        assert db.get_conversation(user["id"], "rec1", context="error") == []
        assert len(db.get_conversation(user["id"], "rec1", context="think")) == 1

    def test_conversations_isolated_between_users(self, temp_db):
        db = temp_db
        u1 = db.register_user("xena")
        u2 = db.register_user("yara")
        db.save_message(u1["id"], "rec1", "user", "u1 message")
        assert db.get_conversation(u2["id"], "rec1") == []

    def test_conversations_isolated_between_problems(self, temp_db):
        db = temp_db
        user = db.register_user("zoe")
        db.save_message(user["id"], "rec1", "user", "about rec1")
        assert db.get_conversation(user["id"], "pat1") == []


class TestExperience:

    def test_increment_category_starts_at_one(self, temp_db):
        db = temp_db
        user = db.register_user("alex")
        db.increment_category(user["id"], "TYPE_ERROR")
        exp = db.get_experience(user["id"])
        assert exp["TYPE_ERROR"] == 1

    def test_increment_category_accumulates(self, temp_db):
        db = temp_db
        user = db.register_user("beth")
        for _ in range(5):
            db.increment_category(user["id"], "SCOPE_ERROR")
        exp = db.get_experience(user["id"])
        assert exp["SCOPE_ERROR"] == 5

    def test_multiple_categories_tracked_independently(self, temp_db):
        db = temp_db
        user = db.register_user("carl")
        db.increment_category(user["id"], "TYPE_ERROR")
        db.increment_category(user["id"], "TYPE_ERROR")
        db.increment_category(user["id"], "PATTERN_MATCH")
        exp = db.get_experience(user["id"])
        assert exp["TYPE_ERROR"]    == 2
        assert exp["PATTERN_MATCH"] == 1

    def test_get_experience_empty_for_new_user(self, temp_db):
        db = temp_db
        user = db.register_user("diana")
        exp = db.get_experience(user["id"])
        assert exp == {}

    def test_experience_isolated_between_users(self, temp_db):
        db = temp_db
        u1 = db.register_user("evan")
        u2 = db.register_user("fiona")
        db.increment_category(u1["id"], "TYPE_ERROR")
        exp2 = db.get_experience(u2["id"])
        assert "TYPE_ERROR" not in exp2



class TestUserStats:

    def test_stats_for_new_user(self, temp_db):
        db = temp_db
        user = db.register_user("george")
        stats = db.get_user_stats(user["id"])
        assert stats["solved"]        == 0
        assert stats["in_progress"]   == 0
        assert stats["messages_sent"] == 0
        assert stats["experience"]    == {}

    def test_stats_reflect_activity(self, temp_db):
        db = temp_db
        user = db.register_user("helen")
        db.mark_problem_solved(user["id"], "rec1")
        db.mark_problem_solved(user["id"], "rec2")
        db.save_code(user["id"], "pat1", "partial")
        db.save_message(user["id"], "rec1", "user", "my answer")
        db.save_message(user["id"], "rec1", "assistant", "next question")
        db.increment_category(user["id"], "TYPE_ERROR")
        stats = db.get_user_stats(user["id"])
        assert stats["solved"]        == 2
        assert stats["in_progress"]   == 1
        assert stats["messages_sent"] == 2
        assert stats["experience"]["TYPE_ERROR"] == 1



class TestDBInit:

    def test_init_db_is_idempotent(self, temp_db):
        """Calling init_db multiple times should not raise or corrupt data."""
        db = temp_db
        user = db.register_user("ian")
        db.init_db()  
        db.init_db()  
        found = db.get_user_by_username("ian")
        assert found is not None

    def test_get_conn_returns_connection(self, temp_db):
        db = temp_db
        conn = db.get_conn()
        assert conn is not None
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1
        conn.close()