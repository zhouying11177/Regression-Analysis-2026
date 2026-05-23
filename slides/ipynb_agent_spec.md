# Notebook Authoring Spec for Agents

## Status
- document_type: notebook_authoring_spec
- scope: `slides/` subtree
- target_files:
  - `slides/weekXX_new/*.ipynb`
  - `slides/weekXX_new/*.py` (Jupytext percent format)
- primary_audience: AI agents
- secondary_audience: human instructors
- source_of_truth: `.py` Jupytext file
- derived_artifact: `.ipynb`
- python_env: `slides/.venv`

## Core Rule
在 `slides/` 下编写课程 notebook 时，**优先维护 Jupytext percent-format 的 `.py` 文件**，再由该 `.py` 文件同步生成 `.ipynb`。

Do:
- edit: `weekXX_class.py`
- generate/sync: `weekXX_class.ipynb`

Do not:
- 只改 `.ipynb` 而不回写 `.py`
- 把 `.ipynb` 当作唯一真实来源

## Rationale
使用 `.py` 作为 source of truth 的原因：
- 更适合 Git diff；
- 更适合 AI 生成和重写；
- 更适合批量重构 notebook 结构；
- 更容易显式维护 cell tags 和教学结构。

## Required Tooling
Assume the `slides/` project uses:
- `uv`
- `jupyter`
- `jupytext`
- `ipykernel`
- `nbformat`

## Directory Convention
For each week notebook package, use:

```text
slides/
├── pyproject.toml
├── .venv/
├── ipynb_agent_spec.md
└── weekXX_new/
    ├── figures/
    ├── weekXX_class.py
    └── weekXX_class.ipynb
```

## File Naming Convention
Preferred names:
- source file: `week12_class.py`
- notebook file: `week12_class.ipynb`

If there are multiple notebooks in the same week, use explicit suffixes:
- `week12_class_main.py`
- `week12_class_demo.py`
- `week12_lab.py`

## Required Notebook Format
The `.py` source file should use Jupytext percent format.

Expected header pattern:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Teaching Model
Each notebook is a classroom artifact, not just a code scratchpad.

Notebook responsibilities:
1. carry the classroom flow;
2. trigger questions before explanations;
3. show phenomena through plots/tables/code execution;
4. preserve teacher-only script notes when needed;
5. keep text short and stage-like.

## Cell Taxonomy
Every major notebook should use these cell roles.

### 1. `meta`
Purpose:
- notebook title
- week number
- learning goals
- usage note

Typical format:
- markdown cell

### 2. `cue`
Purpose:
- ask a question before running code
- prompt prediction
- create tension

Typical format:
- markdown cell

### 3. `stage`
Purpose:
- produce the main classroom event
- code + figure/table/output

Typical format:
- code cell

### 4. `explain`
Purpose:
- give a short interpretation after the output appears
- connect phenomenon to concept

Typical format:
- markdown cell

### 5. `checkpoint`
Purpose:
- force students to make a judgment
- insert a pause for discussion or mini reflection

Typical format:
- markdown cell

### 6. `script`
Purpose:
- teacher-only speaking notes
- reminders about pacing, likely confusion points, or follow-up prompts

Typical format:
- markdown cell
- should include tag `teacher-only`
- should preferably be hidden/collapsible in the frontend

### 7. `backup`
Purpose:
- optional extensions
- extra experiments
- alternative explanation branches

Typical format:
- markdown or code cell

## Required Tags
Use tags consistently.

Minimum tag set:
- `meta`
- `cue`
- `stage`
- `explain`
- `checkpoint`
- `script`
- `teacher-only`
- `backup`

Examples:
- markdown prompt cell: `tags=["cue"]`
- teacher note cell: `tags=["script", "teacher-only"]`
- main demo cell: `tags=["stage"]`
- optional appendix cell: `tags=["backup"]`

## Structure Rule
A notebook section should usually follow this pattern:

```text
cue -> stage -> explain -> checkpoint
```

Not every section must include all four, but this is the default teaching rhythm.

## Content Density Rule
- 每个 markdown cell 只承载一个主要功能；
- 每个 cue cell 最多 1~3 个问题；
- 每个 explain cell 尽量控制在 3~6 行；
- 每个 stage cell 尽量只产生一个主要图或一个主要表；
- 如果代码太长，应拆分为“setup cell + stage cell”。

## Plotting Rule
For classroom plots:
- prefer one strong figure over many small ones;
- set titles and axis labels explicitly;
- annotate the interpretation if needed;
- avoid tiny unreadable defaults;
- prefer deterministic random seeds.

## Data / Reproducibility Rule
- always set random seed in demo notebooks;
- keep datasets lightweight unless explicitly needed;
- do not require secret credentials;
- do not depend on network downloads during class unless explicitly planned.

## Output Rule
Default policy:
- commit the `.ipynb` file for convenience;
- do not rely on embedded outputs as the only source of truth;
- prefer notebooks that can be re-executed from top to bottom.

If not explicitly requested:
- avoid embedding excessive binary output;
- avoid extremely large notebook outputs.

## Teacher-Only Script Rule
Teacher notes should:
- be short;
- focus on pacing or prompting;
- not duplicate the visible explanation cells;
- be tagged with `teacher-only`.

Example content:
- "先让学生猜 20 秒，不要立刻运行。"
- "如果学生答不出 variance，就先说‘模型不稳定’。"
- "这里强调：训练误差下降不代表泛化能力提高。"

## Preferred Week Notebook Skeleton
Recommended high-level structure:

```text
00_meta
01_opening_question
02_core_demo_a
03_core_demo_b
04_metric_or_model_comparison
05_discussion_checkpoints
06_summary
99_backup
```

## Commands
### Create/update notebook from source
From repository root:

```bash
uv run --directory slides jupytext --sync week12_new/week12_class.py
```

### Convert source file into notebook explicitly
```bash
uv run --directory slides jupytext --to ipynb week12_new/week12_class.py -o week12_new/week12_class.ipynb
```

### Execute notebook for validation
```bash
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace week12_new/week12_class.ipynb
```

## Agent Checklist
Before finishing notebook generation, an agent should verify:
1. Is the `.py` file present and readable as the source of truth?
2. Does the notebook use the required cell tags?
3. Is the narrative structured around `cue -> stage -> explain -> checkpoint`?
4. Are teacher-only notes separated from student-visible explanation?
5. Does the notebook run top-to-bottom without external hidden state?
6. Has the `.ipynb` been synced from the `.py` source?
7. Are plots/tables legible enough for classroom projection?

## Week 12 Specific Guidance
For `bias-variance tradeoff` notebooks, the preferred dramatic sequence is:
- ask students to predict what happens as model complexity increases;
- show underfitting vs overfitting on the same synthetic data;
- repeat sampling to show variance;
- inject outliers to compare `RMSE` and `MAE`;
- end with judgment/discussion prompts, not long prose.
