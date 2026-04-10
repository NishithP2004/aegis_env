---
title: Aegis Env Environment Server
emoji: 🎓
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

**Automated Evaluation & Grading Intelligent System** — an OpenEnv environment where an agent grades a student answer by navigating a multi-step evaluation pipeline (`arbiter -> scrutinizer -> validator -> mentor`). After `reset()`, the agent takes multiple `step()` actions. Intermediate steps provide small dense rewards, and the final step yields a nuanced payout. The total episode sum is bounded strictly to **`[0.0, 1.0]`**.

## Quick Start

Use the `AegisEnv` client against a running server. One episode is **reset → step → … → step**; rewards sum to **`[0, 1]`** per [Reward function](#reward-function).

```python
from aegis_env import AegisAction, AegisEnv

with AegisEnv(base_url="http://localhost:8000/openenv").sync() as env:
    r0 = env.reset()
    obs = r0.observation
    r1 = env.step(
        AegisAction(
            proposed_score=4.0,
            agent_reasoning="Meets most rubric criteria with minor gaps; see pipeline history.",
            routing_decision="proceed",
        )
    )
    print(r1.reward, r1.observation.grading_info)
```

The OpenEnv HTTP/WebSocket API is mounted at **`/openenv`** (e.g. `POST /openenv/reset`, `POST /openenv/step` with body `{ "action": { ... } }`).

For an LLM-driven loop and hackathon-style logging, see `inference.py`.

## Web interface

After `uvicorn server.app:app` (or Docker / Hugging Face Spaces), the top-level app serves:

| Path | Purpose |
|------|---------|
| **`/`** | Redirects to **`/web`**. |
| **`/web`** | Custom playground: manual **reset / step**, task difficulty (easy / medium / hard / all), **auto-run** (same prompt loop as `inference.py`), reward chart, reward-function help. |
| **`/web/benchmark`** | **Model benchmark**: list models from an OpenAI-compatible **`GET …/v1/models`** endpoint (default `https://ollama.com/v1`), pick five distinct models + task difficulty, run episodes where **only the chat `model` name** changes, then view tables and Chart.js visualizations. |
| **`/openenv`** | Default OpenEnv UI + HTTP API + WebSocket (same env contract as clients above). |

Static assets live under **`/web/assets/`**. Register explicit `/web` and `/web/benchmark` routes **before** mounting `StaticFiles` at `/web` so HTML routes are not swallowed by the static mount.

### LLM proxy (playground auto-run & server-side helpers)

These endpoints expect API credentials in the server environment (or a benchmark request body where noted):

| Variable | Role |
|----------|------|
| **`HF_TOKEN`**, **`API_KEY`**, or **`OPENAI_API_KEY`** | API key for OpenAI-compatible chat completions. |
| **`API_BASE_URL`** | Chat base URL (default in code may point at Hugging Face Router or your provider). |
| **`MODEL_NAME`** | Default model id when the client does not override it. |

- **`POST /api/llm/complete`** — Proxies chat completions for the playground auto-run (same idea as `inference.py`).

### Stateful HTTP API (custom UI)

The mounted OpenEnv HTTP **`/openenv/step`** contract can be awkward for a thin browser client. The app also exposes:

- **`POST /api/env/reset`** — JSON body may include `task_name` (`easy` \| `medium` \| `hard`), optional `seed`, `episode_id`.
- **`POST /api/env/step`** — JSON body: `{ "action": { "proposed_score", "agent_reasoning", "routing_decision" } }`.

These use a dedicated in-process environment instance for the `/web` UI.

### Benchmark API

- **`GET /api/benchmark/models?api_root=…`** — Lists model ids from **`GET {api_root}/models`** (OpenAI-compatible `data[]` shape; optional fallbacks for other JSON shapes).
- **`POST /api/benchmark/run`** — Body: `models` (1–5 unique names), `task_name`, `max_steps`, optional `seed`, `api_base_url` (OpenAI-compatible chat base for all runs), optional `api_key`. Runs the same episode logic as `inference.py` per model; only the **`model`** field in chat completions varies. Uses a separate `AegisEnvironment` instance from the playground env.

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
- **Custom playground** at **`/web`** and **benchmark** at **`/web/benchmark`**
- **OpenEnv API** at **`/openenv`** (default UI + HTTP + WebSocket)
- **API Documentation** at **`/docs`** on the top-level FastAPI app (when enabled)

## Environment Details

### Action (`AegisAction`)
- `proposed_score` (float) — Proposed score (in `[0, max_score]`; used for accuracy reward).
- `agent_reasoning` (str) — Stage-specific reasoning / critique / feedback text.
- `routing_decision` (str) — Must be `proceed` or `revise` (only meaningful in the `validator` stage).

### Observation (`AegisObservation`)
- `question`, `rubric`, `max_score`, `student_answer` — Prompt shown to the agent (human ground truth is not revealed).
- `current_stage` — Current stage in the pipeline (`arbiter`, `scrutinizer`, `validator`, `mentor`, `finished`).
- `refinement_loops_taken` — Number of validator-requested revision loops taken so far.
- `pipeline_history` — Accumulated pipeline transcript across stages (includes reward history).
- `done` — `False` after `reset()`, `True` once the pipeline completes (or on fatal error).
- `reward` — Dense reward for the transition (final payout occurs at `mentor`).
- `grading_info` — On completion, includes deterministic breakdown (accuracy/validity/flow bank payout).

## Reward function

Rewards are computed inside `server/aegis_env_environment.py` with **no external API calls**. The episode uses a **Flow Bank** initialized at `0.10` to provide dense rewards for following the multi-step pipeline, summing to a maximum of `1.0`.

### Intermediate Steps (Dense Rewards)
- Valid forward transitions (`arbiter` -> `scrutinizer`, etc.) grant **+0.02**.
- A valid `revise` loop (`validator` -> `scrutinizer`) grants **+0.01**.
- These intermediate rewards are deducted from the Flow Bank.
- **Fatal Error Penalty:** Bypassing the sequence, outputting an invalid routing decision, or exceeding the maximum refinement loops (2) instantly terminates the episode with **0.0** reward.

### Final Step (Mentor Payout)
When the agent reaches the `mentor` stage, it receives a final payout combining three metrics:

**Component A — Accuracy (up to 0.6)**
If the `proposed_score` is within valid bounds:
`accuracy_reward = 0.6 × (1 − |norm_agent − norm_human|)`

**Component B — Feedback Validity (up to 0.3)**
If the `agent_reasoning` is \(\ge\) 10 words, it is compared against human reference feedback using Jaccard similarity:
`feedback_reward = 0.3 × Jaccard(agent_text, reference_feedback)`

**Component C — Flow Bank (up to 0.1)**
The remaining balance of the Flow Bank is added to the score. If the agent looped too many times, this balance acts as a compute penalty.

**Total Episode Score = Sum of intermediate rewards + clip₀¹(Accuracy + Validity + Remaining Flow Bank)**

   **feedback_reward = 0.3 × Jaccard(agent_text, reference_feedback)**

### Total

**reward = clip₀¹(accuracy_reward + feedback_reward)**

where `clip₀¹(x) = min(1.0, max(0.0, x))`.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Aegis Env environment server running, point the client at the **OpenEnv mount**:

```python
from aegis_env import AegisAction, AegisEnv

client = AegisEnv(base_url="<ENV_HTTP_URL_HERE>/openenv").sync()
with client:
    r0 = client.reset()
    r1 = client.step(
        AegisAction(
            proposed_score=7.0,
            agent_reasoning="…",
            routing_decision="proceed",
        )
    )
```

Note: When connecting to an existing server, `client.close()` does not stop the server.

### Using the synchronous client wrapper

```python
from aegis_env import AegisAction, AegisEnv

with AegisEnv(base_url="http://localhost:8000/openenv").sync() as env:
    r0 = env.reset()
    print(r0.observation.question[:80], "…")
    r1 = env.step(
        AegisAction(
            proposed_score=5.0,
            agent_reasoning="Adequate but incomplete versus rubric.",
            routing_decision="proceed",
        )
    )
    print(r1.reward, r1.done)
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The OpenEnv app is created in `server/app.py` with `max_concurrent_envs` (default `1`). To allow more simultaneous WebSocket sessions, increase that value in the `create_app(...)` call.

Then multiple clients can connect simultaneously (each session runs its own **reset → step** grading episode):

```python
from aegis_env import AegisAction, AegisEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with AegisEnv(base_url="http://localhost:8000/openenv").sync() as env:
        env.reset()
        result = env.step(
            AegisAction(
                proposed_score=6.0,
                agent_reasoning=f"Client {client_id}: rubric-aligned summary.",
                routing_decision="proceed",
            )
        )
        return client_id, result.reward

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct environment testing

Run the in-process LLM loop (see `inference.py --help` for flags and environment variables such as `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`):

```bash
uv run python inference.py --local
```

Or import `AegisEnvironment` from `server.aegis_env_environment`, call `reset()` then `step(AegisAction(...))`, and inspect `observation.reward` and `observation.grading_info`.

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000/web`** for the playground and **`http://localhost:8000/web/benchmark`** for the benchmark UI.

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
    ├── app.py             # FastAPI app: /web, /api/*, mounts /openenv
    ├── benchmark.py       # Benchmark episode runner (shared prompt path as inference)
    ├── Dockerfile         # Container image definition
    └── web/                 # Static UI (playground + benchmark)
        ├── index.html
        ├── benchmark.html
        └── assets/
            ├── app.js
            └── benchmark.js
```
