#!/usr/bin/env python3
"""Context re-derivation frequency (RDF): a Tier 0 measurement on public agent trajectories.

RDF is the share of a task's input tokens that the agent had already sent earlier in the
same task:

    RDF = re-sent input tokens / all input tokens        (one value per task)

Data: nebius/SWE-rebench-openhands-trajectories, 67,074 OpenHands v0.54.0 runs of
Qwen3-Coder-480B-A35B-Instruct on real GitHub issues (CC-BY-4.0). The OpenHands config
published with the dataset uses the no-op condenser with history truncation off, so every
model call received the full history. This script replays each trajectory that way:
Python counts the tokens in each message, and DuckDB SQL (sql/rdf.sql) turns those counts
into the metric.

Usage
    python rdf_tier0.py selfcheck                # the SQL reproduces the closed form on synthetic loops
    python rdf_tier0.py run                      # full run: download (2.08 GB), count, compute, chart
    python rdf_tier0.py run --sample-frac 0.05   # quick trial on a 5% random sample
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

DATASET = "nebius/SWE-rebench-openhands-trajectories"
DATA_FILE = "trajectories.parquet"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
HERE = Path(__file__).resolve().parent
DEFAULT_SQL = HERE / "sql" / "rdf.sql"
TOKENIZERS = {
    "o200k": "o200k_base (tiktoken)",
    "qwen": "Qwen3-Coder-480B-A35B-Instruct (the generating model's own tokenizer)",
    "chars": "characters (no tokenizer)",
}


def log(*args) -> None:
    print(*args, flush=True)


# ---------------------------------------------------------------------------- text


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def message_text(msg: dict) -> str:
    """The text one message puts in front of the model: its content plus any tool calls."""
    parts = []
    if msg.get("content"):
        parts.append(_as_text(msg["content"]))
    for call in msg.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        parts.append(f"{fn.get('name') or ''}({_as_text(fn.get('arguments'))})")
    return "\n".join(parts)


def _drop_nulls(value):
    """Parquet stores the tool definitions as one wide struct; absent fields come back as None."""
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def tools_text(tools) -> str:
    """The tool definitions as JSON, as sent alongside every model call."""
    if not tools:
        return ""
    if isinstance(tools, str):
        return tools
    return json.dumps(_drop_nulls(tools), ensure_ascii=False)


# ---------------------------------------------------------------------------- tokens


def make_counter(name: str):
    """Return a function that maps a list of texts to a list of token counts."""
    if name == "o200k":
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        threads = os.cpu_count() or 1
        return lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts, num_threads=threads)]
    if name == "qwen":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained("Qwen/Qwen3-Coder-480B-A35B-Instruct")
        return lambda texts: [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
    if name == "chars":
        return lambda texts: [len(t) for t in texts]
    raise ValueError(f"unknown tokenizer {name!r}; choose from {sorted(TOKENIZERS)}")


# ---------------------------------------------------------------------------- pipeline


def download(data_dir: str | Path = "data") -> Path:
    """Fetch the single 2.08 GB parquet file from Hugging Face (no account needed)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", local_dir=str(data_dir))
    log(f"Dataset ready: {path} ({Path(path).stat().st_size / 1e9:.2f} GB)")
    return Path(path)


def extract(
    parquet_path: str | Path,
    work_dir: str | Path = "work",
    tokenizer: str = "o200k",
    sample_frac: float = 1.0,
    seed: int = 7,
    max_tasks: int | None = None,
    batch_size: int = 64,
    flush_rows: int = 1_000_000,
) -> dict:
    """Count tokens per message and write two tables for the SQL step.

    work/messages/part-*.parquet  one row per message: traj_id, idx, role, tokens
    work/task_meta.parquet        one row per trajectory, with the size of its tool definitions
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    work = Path(work_dir)
    msg_dir = work / "messages"
    if msg_dir.exists():
        shutil.rmtree(msg_dir)
    msg_dir.mkdir(parents=True)

    count = make_counter(tokenizer)
    rng = random.Random(seed)
    pf = pq.ParquetFile(str(parquet_path))
    total = pf.metadata.num_rows
    wanted = ["trajectory_id", "instance_id", "repo", "trajectory", "tools", "exit_status", "resolved"]
    columns = [c for c in wanted if c in pf.schema_arrow.names]

    meta = {k: [] for k in ("traj_id", "trajectory_id", "instance_id", "repo", "resolved", "exit_status", "tools_tokens")}
    buf = {"traj_id": [], "idx": [], "role": [], "tokens": []}
    tools_cache: dict[str, int] = {}
    part = seen = kept = n_messages = n_tokens = 0
    started = last_log = time.time()

    def flush():
        nonlocal part
        if not buf["traj_id"]:
            return
        table = pa.table(
            {
                "traj_id": pa.array(buf["traj_id"], pa.int64()),
                "idx": pa.array(buf["idx"], pa.int32()),
                "role": pa.array(buf["role"], pa.string()),
                "tokens": pa.array(buf["tokens"], pa.int64()),
            }
        )
        pq.write_table(table, msg_dir / f"part-{part:05d}.parquet")
        part += 1
        for v in buf.values():
            v.clear()

    log(f"Counting tokens with {TOKENIZERS[tokenizer]}; sample fraction {sample_frac:g} of {total:,} tasks")
    done = False
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        texts, ids, idxs, roles = [], [], [], []
        for row in batch.to_pylist():
            traj_id = seen  # row number in the file: stable across runs
            seen += 1
            if sample_frac < 1.0 and rng.random() >= sample_frac:
                continue
            if max_tasks is not None and kept >= max_tasks:
                done = True
                break
            kept += 1
            trajectory = row.get("trajectory") or []
            if isinstance(trajectory, str):
                trajectory = json.loads(trajectory)
            for i, msg in enumerate(trajectory):
                texts.append(message_text(msg))
                ids.append(traj_id)
                idxs.append(i)
                roles.append((msg.get("role") or "").lower())
            tt = tools_text(row.get("tools"))
            if tt not in tools_cache:
                tools_cache[tt] = count([tt])[0] if tt else 0
            meta["traj_id"].append(traj_id)
            meta["trajectory_id"].append(row.get("trajectory_id"))
            meta["instance_id"].append(row.get("instance_id"))
            meta["repo"].append(row.get("repo"))
            meta["resolved"].append(row.get("resolved"))
            meta["exit_status"].append(row.get("exit_status"))
            meta["tools_tokens"].append(tools_cache[tt])
        if texts:
            tokens = count(texts)
            buf["traj_id"] += ids
            buf["idx"] += idxs
            buf["role"] += roles
            buf["tokens"] += tokens
            n_messages += len(texts)
            n_tokens += sum(tokens)
            if len(buf["traj_id"]) >= flush_rows:
                flush()
        now = time.time()
        if now - last_log > 30 or done or seen == total:
            elapsed = now - started
            eta = elapsed / seen * (total - seen) if seen else 0
            log(
                f"  {seen:>6,}/{total:,} rows read · {kept:,} tasks counted · {n_tokens / 1e9:.2f}B tokens"
                f" · {elapsed / 60:.1f} min elapsed · ~{eta / 60:.0f} min left"
            )
            last_log = now
        if done:
            break
    flush()
    pq.write_table(
        pa.table(
            {
                "traj_id": pa.array(meta["traj_id"], pa.int64()),
                "trajectory_id": pa.array(meta["trajectory_id"], pa.string()),
                "instance_id": pa.array(meta["instance_id"], pa.string()),
                "repo": pa.array(meta["repo"], pa.string()),
                "resolved": pa.array(meta["resolved"], pa.int64()),
                "exit_status": pa.array(meta["exit_status"], pa.string()),
                "tools_tokens": pa.array(meta["tools_tokens"], pa.int64()),
            }
        ),
        work / "task_meta.parquet",
    )
    stats = {
        "rows_in_file": total,
        "tasks_counted": kept,
        "messages": n_messages,
        "message_tokens": n_tokens,
        "tool_definition_variants": len(tools_cache),
        "tool_definition_tokens": sorted(set(tools_cache.values())),
        "tokenizer": tokenizer,
        "sample_frac": sample_frac,
        "seed": seed,
        "extract_minutes": round((time.time() - started) / 60, 1),
    }
    log(f"Counted {kept:,} tasks, {n_messages:,} messages, {n_tokens / 1e9:.2f}B message tokens"
        f" in {stats['extract_minutes']} min")
    return stats


def _connect(messages_glob: str, meta_path: str, db_path: str = ":memory:"):
    import duckdb

    con = duckdb.connect(db_path)
    con.execute(f"CREATE OR REPLACE VIEW messages AS SELECT * FROM read_parquet('{messages_glob}')")
    con.execute(f"CREATE OR REPLACE VIEW task_meta AS SELECT * FROM read_parquet('{meta_path}')")
    return con


def compute(work_dir: str | Path = "work", out_dir: str | Path = "results", sql_path: str | Path = DEFAULT_SQL):
    """Run sql/rdf.sql over the token counts; write the per-task and per-call-count tables."""
    work = Path(work_dir).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    con = _connect((work / "messages" / "*.parquet").as_posix(), (work / "task_meta.parquet").as_posix())
    con.execute(Path(sql_path).read_text())
    con.execute(f"COPY (SELECT * FROM tasks ORDER BY traj_id) TO '{(out / 'tasks.parquet').as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY rdf_by_calls TO '{(out / 'rdf_by_calls.csv').as_posix()}' (HEADER)")
    log(f"SQL done in {time.time() - started:.0f} s: {con.execute('SELECT count(*) FROM tasks').fetchone()[0]:,} tasks")
    return con


# ---------------------------------------------------------------------------- report


def _pct(x, digits=1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _big(n) -> str:
    n = float(n)
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= size:
            return f"{n / size:.1f} {word}"
    return f"{n:,.0f}"


def _row(con, sql: str) -> dict:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    values = cur.fetchone()
    return dict(zip(names, values)) if values else {}


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


# Colors from a validated categorical palette (slots 1-3), light and dark steps.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", slots=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", slots=("#3987e5", "#d95926", "#199e70")),
}


def plot(con, out_dir: str | Path = "results", tokenizer: str = "o200k", min_tasks: int = 20) -> list[Path]:
    """Two panels: where the input tokens went (pooled), and RDF by number of model calls."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.ticker import FixedLocator, PercentFormatter

    s = _row(con, "SELECT * FROM summary_overall")
    data = _rows(con, f"SELECT * FROM rdf_by_calls WHERE tasks >= {int(min_tasks)} ORDER BY n_calls")
    if len(data) < 2:
        data = _rows(con, "SELECT * FROM rdf_by_calls ORDER BY n_calls")
    n = [d["n_calls"] for d in data]
    p10 = [d["p10"] for d in data]
    p50 = [d["p50"] for d in data]
    p90 = [d["p90"] for d in data]
    model_x = list(range(n[0], n[-1] + 1))
    model = [1 - 2 / (k + 1) for k in model_x]
    y_lo = max(0.0, math.floor(min(min(p10), min(model)) * 50) / 50)  # round down to 2 points
    y_ticks = [y_lo + i * 0.02 for i in range(int(round((1 - y_lo) / 0.02)) + 1)]
    while len(y_ticks) > 6:
        y_ticks = y_ticks[::2] if y_ticks[-1] in y_ticks[::2] else y_ticks[1::2]
    x_ticks = list(range(int(math.ceil(n[0] / 10) * 10), n[-1] + 1, 10))

    new = 1 - s["rdf_pooled"]
    preamble = s["preamble_resent_share_of_input"]
    history = s["rdf_pooled"] - preamble
    parts = [("Re-sent: earlier steps of the task", history),
             ("Re-sent: fixed preamble (tools, system prompt, task)", preamble),
             ("Sent for the first time", new)]

    W, H, DPI = 8.0, 6.7, 200
    bar_rect = [0.09, 0.772, 0.88, 0.045]
    line_rect = [0.09, 0.12, 0.88, 0.43]
    out = Path(out_dir)
    paths = []
    for mode, c in THEMES.items():
        fig = plt.figure(figsize=(W, H), dpi=DPI)
        fig.patch.set_facecolor(c["surface"])
        renderer = fig.canvas.get_renderer()

        fig.text(0.09, 0.95, "Context re-derivation in coding-agent loops", fontsize=13.5,
                 fontweight="bold", color=c["ink"])
        fig.text(0.09, 0.913, f"{s['tasks']:,} OpenHands trajectories (Qwen3-Coder-480B) · "
                 f"{_big(s['input_tokens_total'])} input tokens sent to the model", fontsize=9, color=c["ink2"])

        # Panel 1: one whole, split into parts.
        fig.text(0.09, 0.84, "Where the input tokens went", fontsize=10, fontweight="bold", color=c["ink"])
        bx = fig.add_axes(bar_rect)
        bx.set_xlim(0, 1)
        bx.set_ylim(0, 1)
        bx.axis("off")
        px_w = bar_rect[2] * W * DPI
        px_h = bar_rect[3] * H * DPI
        gap, radius = 4 / px_w, 8 / px_w
        x0 = 0.0
        handles = []
        for (label, share), color in zip(parts, c["slots"]):
            if share <= 0:
                continue
            width = max(share - gap, 0.0005)
            bx.add_patch(FancyBboxPatch((x0, 0), width, 1, boxstyle=f"round,pad=0,rounding_size={radius}",
                                        mutation_aspect=px_w / px_h, linewidth=0, facecolor=color))
            text = bx.text(x0 + 12 / px_w, 0.5, _pct(share), ha="left", va="center", fontsize=9,
                           fontweight="bold", color="#ffffff")
            if text.get_window_extent(renderer).width + 24 > width * px_w:
                text.remove()  # too narrow to label inside; the legend and table carry it
            handles.append(Patch(facecolor=color, label=f"{label} · {_pct(share)}"))
            x0 += share
        legend = bx.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.35), ncol=1, frameon=False,
                           fontsize=8.3, handlelength=1.0, handleheight=1.0, borderaxespad=0, labelspacing=0.35)
        for text in legend.get_texts():
            text.set_color(c["ink2"])

        # Panel 2: RDF by task length, against the closed form.
        fig.text(0.09, 0.585, "Share of each task's input already sent earlier in that task, by task length",
                 fontsize=10, fontweight="bold", color=c["ink"])
        ax = fig.add_axes(line_rect)
        ax.set_facecolor(c["surface"])
        series = c["slots"][0]
        ax.fill_between(n, p10, p90, color=series, alpha=0.14 if mode == "light" else 0.26, linewidth=0,
                        label="10th–90th percentile of tasks")
        ax.plot(model_x, model, color=c["ink2"], linewidth=1.5, linestyle=(0, (4, 3)),
                label="Uniform-loop model, 1 − 2/(n+1)")
        ax.plot(n, p50, color=series, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                label="Median task")
        ax.plot([n[-1]], [p50[-1]], marker="o", markersize=6, color=series,
                markeredgecolor=c["surface"], markeredgewidth=2)
        ax.annotate(f"Median at {n[-1]} calls: {_pct(p50[-1])}", (n[-1], p50[-1]), xytext=(-8, 8),
                    textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=c["ink"])
        k = min(len(model_x) - 1, 5)
        ax.annotate("Uniform-loop model", (model_x[k], model[k]), xytext=(6, -8), textcoords="offset points",
                    ha="left", va="top", fontsize=8.5, color=c["ink2"])

        ax.set_xlim(n[0] - 1, n[-1] + 2)
        ax.set_ylim(y_lo, 1.0)
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["axis"])
        ax.tick_params(colors=c["muted"], labelsize=8.5, length=0, pad=6)
        ax.set_xlabel("Model calls in the task", color=c["ink2"], fontsize=9, labelpad=8)
        legend = ax.legend(loc="lower right", frameon=False, fontsize=8.3, handlelength=2.2)
        for text in legend.get_texts():
            text.set_color(c["ink2"])

        fig.text(0.09, 0.02, f"Data: {DATASET} (CC-BY-4.0). Tokens: {TOKENIZERS[tokenizer].split(' (')[0]}."
                 f" Axis starts at {y_lo * 100:.0f}%. Call counts with at least {min_tasks} tasks shown.",
                 fontsize=7.3, color=c["muted"])
        path = out / ("rdf_vs_calls.png" if mode == "light" else "rdf_vs_calls_dark.png")
        fig.savefig(path, facecolor=c["surface"])
        plt.close(fig)
        paths.append(path)
    return paths


def report(con, out_dir: str | Path = "results", stats: dict | None = None, runtime_min: float | None = None) -> str:
    """Write results/summary.md and results/summary.json; return the markdown."""
    stats = stats or {}
    out = Path(out_dir)
    s = _row(con, "SELECT * FROM summary_overall")
    by_outcome = _rows(con, "SELECT * FROM summary_by_outcome")
    bands = _rows(con, "SELECT * FROM summary_by_band")
    spread = con.execute("SELECT quantile_cont(p90 - p10, 0.5) FROM rdf_by_calls WHERE tasks >= 50").fetchone()[0]
    tokenizer = stats.get("tokenizer", "o200k")
    plot(con, out, tokenizer=tokenizer)

    gap = s["gap_vs_closed_form_median"]
    higher = "higher" if gap >= 0 else "lower"
    lines = [
        f"- Across {s['tasks']:,} tasks, the agent sent the model {_big(s['input_tokens_total'])} input tokens;"
        f" {_pct(s['rdf_pooled'])} of them had already been sent earlier in the same task.",
        f"- The median task made {s['calls_median']:.0f} model calls and had an RDF of {_pct(s['rdf_median'])}"
        f" (10th–90th percentile {_pct(s['rdf_p10'])}–{_pct(s['rdf_p90'])}).",
        f"- The closed form 1 − 2/(n+1), derived for a loop that adds equal new context per call, comes within"
        f" {abs(gap) * 100:.1f} percentage points of the median task's measured RDF (measured is {higher}).",
    ]
    if spread is not None:
        if spread < 0.01:
            lines.append(
                f"- At a given number of calls RDF barely varies: the 10th and 90th percentile tasks sit"
                f" {spread * 100:.1f} percentage points apart, so loop length alone predicts it."
            )
        else:
            lines.append(
                f"- Tasks with the same number of calls still differ: at a typical call count, the 10th and"
                f" 90th percentile tasks are {spread * 100:.1f} percentage points apart."
            )
    lines.append(
        f"- The fixed preamble (tool definitions, system prompt and task statement, sent with every call)"
        f" accounts for {_pct(s['preamble_share_pooled'])} of all re-sent tokens."
    )
    if len(by_outcome) == 2:
        r = {o["outcome"]: o for o in by_outcome}
        a, b = r.get("resolved"), r.get("not resolved")
        if a and b:
            lines.append(
                f"- Resolved tasks: median {a['calls_median']:.0f} calls, RDF {_pct(a['rdf_median'])};"
                f" not resolved: median {b['calls_median']:.0f} calls, RDF {_pct(b['rdf_median'])}."
            )
    lines.append(
        f"- Modeled cost share (CACS): re-sent input is {_pct(s['cacs_no_cache_pooled'])} of spend with no caching"
        f" and {_pct(s['cacs_prefix_cache_pooled'])} with perfect prefix caching (cache reads at 0.1×, writes at"
        f" 1.25×, output at 5× the fresh-input price). Caching changes the price of a re-sent token, not how"
        f" often it is sent."
    )

    def table(rows, cols):
        head = "| " + " | ".join(c[0] for c in cols) + " |"
        rule = "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"
        body = ["| " + " | ".join(fmt(r[key]) for _, key, fmt in cols) + " |" for r in rows]
        return "\n".join([head, rule] + body)

    num = lambda v: f"{v:,.0f}"
    run_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    md = "\n".join(
        [
            "# Results: context re-derivation frequency, Tier 0",
            "",
            f"Run {run_date} (UTC) · {s['tasks']:,} of {stats.get('rows_in_file', s['tasks']):,} tasks measured"
            f" · sample fraction {stats.get('sample_frac', 1.0):g} (seed {stats.get('seed', 7)})"
            f" · tokens: {TOKENIZERS[tokenizer]}"
            + (f" · runtime {runtime_min:.0f} min" if runtime_min else ""),
            "",
            "## Paste into README → Results",
            "",
            *lines,
            "",
            "## By outcome",
            "",
            table(by_outcome, [("Outcome", "outcome", str), ("Tasks", "tasks", num),
                               ("Median calls", "calls_median", num), ("Median RDF", "rdf_median", _pct),
                               ("p10", "rdf_p10", _pct), ("p90", "rdf_p90", _pct),
                               ("Median input tokens", "input_tokens_median", num)]),
            "",
            "## By task length",
            "",
            table(bands, [("Model calls", "calls_band", str), ("Tasks", "tasks", num),
                          ("Median RDF", "rdf_median", _pct), ("p10", "rdf_p10", _pct), ("p90", "rdf_p90", _pct),
                          ("Uniform-loop model", "closed_form_median", _pct)]),
            "",
            "## Checks",
            "",
            f"- RDF counting messages only (tool definitions left out): median {_pct(s['rdf_messages_only_median'])}.",
            f"- Tool definitions sent with each call: {', '.join(f'{t:,}' for t in stats.get('tool_definition_tokens', []))} tokens.",
            f"- Per-call-count table: `rdf_by_calls.csv`. Per-task table: `tasks.parquet`.",
            "",
            "![Context re-derivation frequency by task length](rdf_vs_calls.png)",
            "",
        ]
    )
    (out / "summary.md").write_text(md)
    payload = {"summary": s, "by_outcome": by_outcome, "by_band": bands, "within_call_count_spread_p10_p90": spread,
               "run": {**stats, "runtime_minutes": runtime_min, "date_utc": run_date}}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    log(f"Wrote {out / 'summary.md'}, summary.json and the charts")
    return md


# ---------------------------------------------------------------------------- self-check


def self_check(sql_path: str | Path = DEFAULT_SQL, verbose: bool = True) -> None:
    """Run the SQL on synthetic append-only loops whose answer is known in closed form.

    A task with n calls where every call adds k new tokens has RDF = 1 - 2/(n+1). With a
    tool block of T tokens sent on every call, RDF = 1 - (T + n*k) / (n*T + k*n*(n+1)/2).
    Each synthetic trajectory also ends with a tool message that is never sent, which must
    count nowhere.
    """
    import duckdb
    import pyarrow as pa

    k, a = 1000, 200  # new tokens per call; the assistant's share of them
    msgs = {"traj_id": [], "idx": [], "role": [], "tokens": []}
    meta = {"traj_id": [], "trajectory_id": [], "instance_id": [], "repo": [], "resolved": [],
            "exit_status": [], "tools_tokens": []}
    expected = {}
    traj_id = 0
    for tools in (0, 4000):
        for n in range(1, 101):
            seq = [("system", k)] + [m for _ in range(n) for m in (("assistant", a), ("tool", k - a))]
            for i, (role, tokens) in enumerate(seq):
                msgs["traj_id"].append(traj_id)
                msgs["idx"].append(i)
                msgs["role"].append(role)
                msgs["tokens"].append(tokens)
            for key, value in (("traj_id", traj_id), ("trajectory_id", f"t{traj_id}"), ("instance_id", "synthetic"),
                               ("repo", "synthetic"), ("resolved", n % 2), ("exit_status", "submit"),
                               ("tools_tokens", tools)):
                meta[key].append(value)
            expected[traj_id] = 1 - (tools + n * k) / (n * tools + k * n * (n + 1) / 2)
            traj_id += 1
    con = duckdb.connect()
    con.register("messages_src", pa.table(msgs))
    con.register("task_meta_src", pa.table(meta))
    con.execute("CREATE VIEW messages AS SELECT * FROM messages_src")
    con.execute("CREATE VIEW task_meta AS SELECT * FROM task_meta_src")
    con.execute(Path(sql_path).read_text())
    got = dict(con.execute("SELECT traj_id, rdf FROM tasks").fetchall())
    closed = dict(con.execute("SELECT traj_id, rdf_closed_form FROM tasks").fetchall())
    worst = max(abs(got[t] - expected[t]) for t in expected)
    worst_closed = max(abs(got[t] - closed[t]) for t in range(100))  # the no-tools half
    assert len(got) == len(expected), f"expected {len(expected)} tasks, got {len(got)}"
    assert worst < 1e-12, f"RDF differs from the closed form by {worst}"
    assert worst_closed < 1e-12, f"rdf_closed_form column differs by {worst_closed}"
    if verbose:
        log(f"Self-check passed: {len(expected)} synthetic loops (1-100 calls, with and without a tool block)"
            f" match the closed form to within {max(worst, worst_closed):.0e}.")


# ---------------------------------------------------------------------------- CLI


def run(data_dir="data", work_dir="work", out_dir="results", tokenizer="o200k", sample_frac=1.0, seed=7,
        max_tasks=None, parquet=None) -> str:
    started = time.time()
    self_check()
    path = Path(parquet) if parquet else download(data_dir)
    stats = extract(path, work_dir, tokenizer=tokenizer, sample_frac=sample_frac, seed=seed, max_tasks=max_tasks)
    con = compute(work_dir, out_dir)
    return report(con, out_dir, stats, runtime_min=(time.time() - started) / 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="check the SQL against the closed form on synthetic loops")
    r = sub.add_parser("run", help="download, count tokens, compute RDF, write results/")
    r.add_argument("--sample-frac", type=float, default=1.0, help="share of tasks to measure (default 1.0)")
    r.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="o200k")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--max-tasks", type=int, default=None)
    r.add_argument("--parquet", default=None, help="use a local copy of trajectories.parquet")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--work-dir", default="work")
    r.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        self_check()
        return 0
    md = run(args.data_dir, args.work_dir, args.out, args.tokenizer, args.sample_frac, args.seed,
             args.max_tasks, args.parquet)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
