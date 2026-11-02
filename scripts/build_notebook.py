"""Rebuild notebooks/rdf_tier0_colab.ipynb from rdf_tier0.py and sql/rdf.sql.

The notebook embeds both files so it runs on Google Colab with nothing else uploaded.
Run this after editing either file:  python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Context re-derivation frequency · Tier 0

How much of what an AI agent sends to the model has it already sent earlier in the same task?
This notebook measures it on **67,074 public coding-agent trajectories**
([nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), CC-BY-4.0):
Python counts the tokens in every message, and DuckDB SQL turns those counts into the metric.

**To run:** *Runtime → Run all.* The full run takes about 30 minutes on Colab's free CPU and needs no account
or API key. For a 3-minute trial first, set `SAMPLE_FRAC = 0.05` below. The last cell downloads
`results.zip`: a summary, two charts and the per-task table."""

PARAMS = """SAMPLE_FRAC = 1.0    # 1.0 = all 67,074 tasks (about 30 min). 0.05 = a 5% trial (about 3 min).
TOKENIZER = "o200k"  # "o200k" (fast) or "qwen" (the generating model's own tokenizer, about 3x slower)"""

SETUP = """!pip install -q duckdb tiktoken huggingface_hub
import os, time
os.makedirs("sql", exist_ok=True)
START = time.time()"""

FILES_MD = """## The pipeline files

The next two cells write `rdf_tier0.py` (Python: download, token counting, charts) and `sql/rdf.sql`
(the metric itself), the same files as in the repository. Collapse them if you only want the results."""

STEPS = [
    ("## Step 0 · Check the SQL against the closed form\n\n"
     "Before touching real data, the SQL runs on 200 synthetic loops whose answer is known exactly: "
     "a loop of *n* calls that each add equal new context has RDF = 1 − 2/(*n*+1).",
     "import importlib, rdf_tier0 as r\nimportlib.reload(r)\nr.self_check()"),
    ("## Step 1 · Download the dataset (2.08 GB)",
     'path = r.download("data")'),
    ("## Step 2 · Count tokens per message (Python)\n\n"
     "Each message's text (content plus any tool call) is tokenized once; the tool definitions sent with "
     "every call are counted once per task. Progress prints every 30 seconds.",
     'stats = r.extract(path, "work", tokenizer=TOKENIZER, sample_frac=SAMPLE_FRAC)\nstats'),
    ("## Step 3 · Compute RDF (DuckDB SQL)\n\n"
     "A window function sums the history in front of every model call; each task's RDF is re-sent input "
     "divided by all input. See `sql/rdf.sql`.",
     'con = r.compute("work", "results")\ncon.execute("SELECT * FROM summary_overall").df().T'),
    ("## Step 4 · Results",
     'from IPython.display import Image, Markdown, display\n'
     'md = r.report(con, "results", stats, runtime_min=(time.time() - START) / 60)\n'
     'display(Image("results/rdf_vs_calls.png", width=900))\n'
     'display(Markdown(md.split("![")[0]))'),
    ("## Step 5 · Download the results\n\n"
     "Unzip `results.zip` and upload the `results` folder to the repository.",
     'import shutil\nshutil.make_archive("results", "zip", "results")\n'
     'try:\n    from google.colab import files\n    files.download("results.zip")\n'
     'except ImportError:\n    print("Not on Colab: see ./results and ./results.zip")'),
]


def build() -> Path:
    code = (ROOT / "rdf_tier0.py").read_text()
    sql = (ROOT / "sql" / "rdf.sql").read_text()
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(INTRO),
        nbf.v4.new_code_cell(PARAMS),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell(FILES_MD),
        nbf.v4.new_code_cell("%%writefile rdf_tier0.py\n" + code),
        nbf.v4.new_code_cell("%%writefile sql/rdf.sql\n" + sql),
    ]
    for md, src in STEPS:
        cells += [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(src)]
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    out = ROOT / "notebooks" / "rdf_tier0_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    print(build())
