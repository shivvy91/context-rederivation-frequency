"""Run with:  pytest -q"""
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rdf_tier0 as r  # noqa: E402

SQL = Path(__file__).resolve().parents[1] / "sql" / "rdf.sql"


def run_sql(messages, meta):
    con = duckdb.connect()
    con.register("m", pa.table(messages))
    con.register("t", pa.table(meta))
    con.execute("CREATE VIEW messages AS SELECT * FROM m")
    con.execute("CREATE VIEW task_meta AS SELECT * FROM t")
    con.execute(SQL.read_text())
    return con


def one_task(seq, tools=0):
    messages = {"traj_id": [0] * len(seq), "idx": list(range(len(seq))),
                "role": [s[0] for s in seq], "tokens": [s[1] for s in seq]}
    meta = {"traj_id": [0], "trajectory_id": ["x"], "instance_id": ["x"], "repo": ["x"],
            "resolved": [1], "exit_status": ["submit"], "tools_tokens": [tools]}
    return run_sql(messages, meta)


def test_closed_form():
    r.self_check(SQL, verbose=False)


def test_hand_worked_task():
    # calls see 10, then 10+5+20=35, then 35+5+30=70 tokens of history; 100 tokens of tools each time
    con = one_task([("system", 10), ("assistant", 5), ("tool", 20), ("assistant", 5), ("tool", 30),
                    ("assistant", 5), ("tool", 999)], tools=100)
    row = con.execute("SELECT n_calls, input_tokens, new_input_tokens, resent_input_tokens, "
                      "output_tokens, preamble_tokens, preamble_resent_tokens, rdf FROM tasks").fetchone()
    input_tokens = (100 + 10) + (100 + 35) + (100 + 70)  # 415
    new_tokens = 100 + 70  # tools once, plus every message before the last call
    assert row[:7] == (3, input_tokens, new_tokens, input_tokens - new_tokens, 15, 110, 220)
    assert abs(row[7] - (input_tokens - new_tokens) / input_tokens) < 1e-12


def test_consecutive_assistant_messages():
    con = one_task([("system", 8), ("assistant", 2), ("assistant", 2), ("tool", 4), ("assistant", 2)])
    n_calls, input_tokens, new_tokens = con.execute(
        "SELECT n_calls, input_tokens, new_input_tokens FROM tasks").fetchone()
    assert (n_calls, input_tokens, new_tokens) == (3, 8 + 10 + 16, 16)


def test_message_text_includes_tool_calls():
    msg = {"role": "assistant", "content": "Let me look.",
           "tool_calls": [{"function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}]}
    assert r.message_text(msg) == 'Let me look.\nexecute_bash({"command": "ls"})'
    assert r.message_text({"role": "tool", "content": None}) == ""


def test_tools_text_drops_parquet_nulls():
    tools = [{"type": "function", "function": {"name": "think", "description": "d",
                                               "parameters": {"type": "object", "file_text": None}}}]
    assert "file_text" not in r.tools_text(tools)
