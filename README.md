---
title: Aegis Env Environment Server
emoji: 🔕
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AEGIS-Env (Aegis Env)

**Automated Evaluation & Grading Intelligent System** — an OpenEnv environment where an agent grades a student answer from a question, rubric, and response. After `reset()`, a single `step()` produces a scalar reward in **`[0.0, 1.0]`** using deterministic rules (no LLM inside the environment). See [Reward function](#reward-function) below.

## Quick Start

Use the `AegisEnv` client against a running server (set `MONGO_URI` so the server can load the dataset at startup). One episode is **reset → step**; the step reward is in **`[0, 1]`** per [Reward function](#reward-function).

```python
from aegis_env import AegisAction, AegisEnv

with AegisEnv(base_url="http://localhost:8000").sync() as env:
    r0 = env.reset()
    obs = r0.observation
    r1 = env.step(
        AegisAction(
            final_score=8.0,
            score_justification="Meets most rubric criteria with minor gaps.",
            improvement_advice="Expand the conclusion and cite one more primary source.",
        )
    )
    print(r1.reward, r1.observation.grading_info)
```

For an LLM-driven loop and hackathon-style logging, see `inference.py`.

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t aegis_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action (`AegisAction`)
- `final_score` (float) — Assigned score (must lie in `[0, max_score]` to earn the accuracy component; see rewards).
- `score_justification` (str) — Reasoning for the score.
- `improvement_advice` (str) — Actionable feedback for the student.

### Observation (`AegisObservation`)
- `question`, `rubric`, `max_score`, `student_answer` — Prompt shown to the agent (human ground truth is not revealed).
- `done` — `False` after `reset()`, `True` after `step()` (single-step episode).
- `reward` — Set on the post-step observation; `None` at reset.
- `grading_info` — After `step()`, contains deterministic breakdown: `accuracy_reward`, `feedback_reward`, `total_reward`, `word_count`.

## Reward function

Rewards are computed entirely inside `server/aegis_env_environment.py` with **no external API calls**. The returned value is **clipped to `[0.0, 1.0]`**.

### Component A — Accuracy (up to **0.7**)

Let `obtained_score` be the human reference score for the current row (loaded from MongoDB at startup, kept in server state after `reset()`). Let `max_score` be the item cap from the same row.

1. Normalize: `norm_human = obtained_score / max_score`, `norm_agent = final_score / max_score`.
2. If `max_score ≤ 0`, accuracy reward is **0**.
3. If `final_score < 0` or `final_score > max_score`, accuracy reward is **0** (out-of-range penalty).
4. Otherwise:

   **accuracy_reward = 0.7 × (1 − |norm_agent − norm_human|)**

So a perfect match to the human score gives **0.7** from this term.

### Component B — Feedback validity (up to **0.3**)

1. Concatenate: `agent_text = score_justification + " " + improvement_advice` (trimmed).
2. If `agent_text` has **fewer than 10 words** (whitespace split), **feedback_reward = 0**.
3. Otherwise compute **Jaccard similarity** between the word sets of `agent_text` and the stored **reference feedback** (same record as the human score; simple `lower().split()` tokenization, set intersection / union).

   **feedback_reward = 0.3 × Jaccard(agent_text, reference_feedback)**

### Total

**reward = clip₀¹(accuracy_reward + feedback_reward)**

where `clip₀¹(x) = min(1.0, max(0.0, x))`.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Aegis Env environment server running, you can connect directly:

```python
from aegis_env import AegisAction, AegisEnv

client = AegisEnv(base_url="<ENV_HTTP_URL_HERE>").sync()
with client:
    r0 = client.reset()
    r1 = client.step(
        AegisAction(
            final_score=7.0,
            score_justification="…",
            improvement_advice="…",
        )
    )
```

Note: When connecting to an existing server, `client.close()` does not stop the server.

### Using the synchronous client wrapper

```python
from aegis_env import AegisAction, AegisEnv

with AegisEnv(base_url="http://localhost:8000").sync() as env:
    r0 = env.reset()
    print(r0.observation.question[:80], "…")
    r1 = env.step(
        AegisAction(
            final_score=5.0,
            score_justification="Adequate but incomplete versus rubric.",
            improvement_advice="Address criterion 2 explicitly and add an example.",
        )
    )
    print(r1.reward, r1.done)
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    AegisEnvironment,  # Pass class, not instance
    AegisAction,
    AegisObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously (each session runs its own **reset → step** grading episode):

```python
from aegis_env import AegisAction, AegisEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with AegisEnv(base_url="http://localhost:8000").sync() as env:
        env.reset()
        result = env.step(
            AegisAction(
                final_score=6.0,
                score_justification=f"Client {client_id}: rubric-aligned summary.",
                improvement_advice="Strengthen evidence and proof steps across the board.",
            )
        )
        return client_id, result.reward

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct environment testing

Run the in-process loop (requires `MONGO_URI` and `HF_TOKEN`; see `inference.py --help`):

```bash
uv run python inference.py --local
```

Or exercise the package via a short Python snippet after setting `MONGO_URI`, importing `AegisEnvironment`, calling `reset()` then `step(AegisAction(...))`, and inspecting `observation.reward` and `observation.grading_info`.

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
aegis_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # AegisEnv client
├── models.py              # Action and Observation models
├── inference.py           # LLM baseline + hackathon logging
└── server/
    ├── __init__.py        # Server module exports
    ├── aegis_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
