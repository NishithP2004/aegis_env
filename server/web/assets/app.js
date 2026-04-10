const API_BASE = "/api/env";
const LLM_API = "/api/llm/complete";

const AUTO_TEMPERATURE = 0.2;
const AUTO_MAX_TOKENS = 4096;

const els = {
  episodeId: document.getElementById("episode-id"),
  stepCount: document.getElementById("step-count"),
  stageName: document.getElementById("stage-name"),

  maxScore: document.getElementById("max-score"),
  maxScoreInline: document.getElementById("max-score-inline"),

  q: document.getElementById("obs-question"),
  rubric: document.getElementById("obs-rubric"),
  student: document.getElementById("obs-student"),
  history: document.getElementById("obs-history"),

  diagCard: document.getElementById("diagnostics-card"),
  diagGrid: document.getElementById("diag-grid"),

  score: document.getElementById("inp-score"),
  scoreMaxBtn: document.getElementById("btn-score-max"),
  reason: document.getElementById("inp-reason"),

  routeBtns: Array.from(document.querySelectorAll(".route-btn")),
  routeSelected: document.getElementById("route-selected"),

  resetBtn: document.getElementById("btn-reset"),
  stepBtn: document.getElementById("btn-step"),
  stepSpinner: document.getElementById("step-spinner"),
  stepLabel: document.getElementById("step-label"),

  errorBanner: document.getElementById("error-banner"),
  errorText: document.getElementById("error-text"),
  errorDismiss: document.getElementById("error-dismiss"),

  taskSelect: document.getElementById("task-select"),
  maxSteps: document.getElementById("max-steps"),
  autoRunBtn: document.getElementById("btn-auto-run"),
  autoStopBtn: document.getElementById("btn-auto-stop"),
  autoStatus: document.getElementById("auto-status"),
  rewardChartCanvas: document.getElementById("reward-chart"),
};

let state = {
  obs: null,
  routingDecision: "proceed",
  episodeId: null,
  stepCount: null,
  loading: false,
  autoRunning: false,
};

let rewardChart = null;
let autoAbort = null;

function setError(err) {
  const msg =
    typeof err === "string"
      ? err
      : err?.stack || err?.message || JSON.stringify(err, null, 2);
  els.errorText.textContent = msg;
  els.errorBanner.classList.remove("hidden");
}

function clearError() {
  els.errorBanner.classList.add("hidden");
  els.errorText.textContent = "";
}

function setLoading(isLoading) {
  state.loading = isLoading;
  const blockManual = isLoading || state.autoRunning;
  els.resetBtn.disabled = blockManual;
  els.stepBtn.disabled =
    blockManual || Boolean(state.obs?.done);

  if (isLoading && !state.autoRunning) {
    els.stepSpinner.classList.remove("hidden");
    els.stepLabel.textContent = "Working…";
  } else if (!state.autoRunning) {
    els.stepSpinner.classList.add("hidden");
    els.stepLabel.textContent = "Take Step";
  }
}

function setAutoRunning(on) {
  state.autoRunning = on;
  els.autoRunBtn.disabled = on;
  els.autoStopBtn.disabled = !on;
  els.taskSelect.disabled = on;
  els.maxSteps.disabled = on;
  els.resetBtn.disabled = on || state.loading;
  els.stepBtn.disabled =
    on || state.loading || Boolean(state.obs?.done);
  if (on) {
    els.stepSpinner.classList.remove("hidden");
    els.stepLabel.textContent = "Auto-run…";
  } else {
    setLoading(false);
  }
}

function extractResult(json) {
  if (json && typeof json === "object") {
    if (json.observation && typeof json.observation === "object") {
      return { obs: json.observation, reward: json.reward, done: json.done };
    }
    return { obs: json, reward: json.reward, done: json.done };
  }
  return { obs: null, reward: null, done: null };
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    // keep null
  }
  if (!res.ok) {
    throw new Error(
      `HTTP ${res.status} ${res.statusText}\n` + (text || "<empty response>")
    );
  }
  return json;
}

/** OpenEnv HTTP `POST /step` expects `{ action: <AegisAction> }`, not a flat body. */
function bodyForStep(action) {
  return { action };
}

function resetBodyForTaskSelect() {
  const v = els.taskSelect.value;
  if (v === "all") return {};
  return { task_name: v };
}

function renderDiagnostics(gradingInfo) {
  els.diagGrid.innerHTML = "";
  if (!gradingInfo || typeof gradingInfo !== "object") return;

  const entries = Object.entries(gradingInfo);
  for (const [k, v] of entries) {
    const card = document.createElement("div");
    card.className =
      "rounded-2xl border border-slate-200 bg-white/70 px-4 py-3 shadow-sm";

    const label = document.createElement("div");
    label.className = "text-xs font-semibold uppercase tracking-wide text-slate-500";
    label.textContent = k;

    const val = document.createElement("div");
    val.className = "mt-1 font-mono text-sm text-slate-900";
    if (typeof v === "number") val.textContent = v.toFixed(4);
    else val.textContent = typeof v === "string" ? v : JSON.stringify(v);

    card.appendChild(label);
    card.appendChild(val);
    els.diagGrid.appendChild(card);
  }
}

function renderObs(obs) {
  state.obs = obs;

  const maxScore = Number(obs?.max_score ?? NaN);
  const stage = obs?.current_stage ?? "—";

  els.stageName.textContent = stage;
  els.q.textContent = obs?.question ?? "—";
  els.rubric.textContent = obs?.rubric ?? "—";
  els.student.textContent = obs?.student_answer ?? "—";
  els.history.textContent = obs?.pipeline_history ?? "—";

  const meta = obs?.metadata && typeof obs.metadata === "object" ? obs.metadata : {};
  els.episodeId.textContent =
    meta.episode_id != null ? String(meta.episode_id) : "—";
  els.stepCount.textContent =
    meta.step_count != null ? String(meta.step_count) : "—";

  if (Number.isFinite(maxScore)) {
    els.maxScore.textContent = maxScore.toString();
    els.maxScoreInline.textContent = maxScore.toString();
    els.score.max = maxScore.toString();
  } else {
    els.maxScore.textContent = "—";
    els.maxScoreInline.textContent = "—";
    els.score.removeAttribute("max");
  }

  const block = state.loading || state.autoRunning;
  els.stepBtn.disabled = block || Boolean(obs?.done);

  if (obs?.done) {
    els.diagCard.classList.remove("hidden");
    renderDiagnostics(obs?.grading_info);
  } else {
    els.diagCard.classList.add("hidden");
    els.diagGrid.innerHTML = "";
  }

  const isValidator = String(stage).toLowerCase() === "validator";
  for (const b of els.routeBtns) b.disabled = !isValidator || block;
}

function setRoutingDecision(value) {
  state.routingDecision = value;
  els.routeSelected.textContent = value;
  for (const b of els.routeBtns) {
    const active = b.dataset.value === value;
    b.classList.toggle("bg-slate-900", active);
    b.classList.toggle("text-white", active);
    b.classList.toggle("border-slate-900", active);
    b.classList.toggle("bg-white/70", !active);
    b.classList.toggle("text-slate-800", !active);
    b.classList.toggle("border-slate-200", !active);
  }
}

function stripMarkdownJsonFence(text) {
  if (typeof text !== "string") return "";
  if (text.startsWith("```")) {
    const lines = text.split("\n");
    const last = lines[lines.length - 1]?.trim() === "```";
    return (last ? lines.slice(1, -1) : lines.slice(1)).join("\n");
  }
  return text;
}

function parseFirstJsonObject(s) {
  const cleaned = String(s).trim();
  let start = cleaned.indexOf("{");
  if (start === -1) throw new Error("No JSON object found in model output");
  let depth = 0;
  let inStr = false;
  let esc = false;
  let q = "";
  for (let j = start; j < cleaned.length; j++) {
    const c = cleaned[j];
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (c === "\\") {
        esc = true;
      } else if (c === q) {
        inStr = false;
        q = "";
      }
      continue;
    }
    if (c === '"' || c === "'") {
      inStr = true;
      q = c;
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) {
        return JSON.parse(cleaned.slice(start, j + 1));
      }
    }
  }
  throw new Error("Unbalanced JSON object in model output");
}

function parseAction(raw, maxScore) {
  const cleaned = stripMarkdownJsonFence(raw).trim();
  const data = parseFirstJsonObject(cleaned);
  const hi = Number.isFinite(maxScore) ? maxScore : 1;
  let ps = Number(data.proposed_score ?? 0);
  if (!Number.isFinite(ps)) ps = 0;
  ps = Math.min(Math.max(ps, 0), hi);
  return {
    proposed_score: ps,
    agent_reasoning: String(data.agent_reasoning ?? ""),
    routing_decision: String(data.routing_decision ?? "proceed"),
  };
}

function repairPromptInvalidOutput(raw, maxScore) {
  const clipped = String(raw).slice(0, 2000);
  return (
    "Your previous response was not valid JSON.\n" +
    "Return ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:\n" +
    "{\n" +
    `  "proposed_score": <number between 0 and ${maxScore}>,\n` +
    '  "agent_reasoning": "<string>",\n' +
    '  "routing_decision": "proceed" or "revise"\n' +
    "}\n\n" +
    "Invalid response to fix:\n" +
    clipped
  );
}

function buildUserPrompt(step, lastAction, lastReward, history, obs) {
  const stage = String(obs?.current_stage ?? "");
  const loops = Number(obs?.refinement_loops_taken ?? 0);
  const maxScore = Number(obs?.max_score ?? 1) || 1.0;

  let persona = "";
  if (stage === "arbiter") {
    persona = `You are **The Arbiter**, the initial routing/assessment agent in the automated grading pipeline.

Goal: Perform the first comprehensive evaluation of the student's answer against the rubric.

Instructions:
- Conduct a thorough preliminary analysis against the rubric.
- Propose an initial score and explain strengths + areas of concern.
- Your reasoning will be used by later stages; be clear and complete.
- Set routing_decision to 'proceed'.`;
  } else if (stage === "scrutinizer") {
    persona = `You are **The Scrutinizer**, the first refinement agent.

Goal: Critically examine and improve the current assessment.

Instructions:
- Review the pipeline history for accuracy, completeness, and rubric alignment.
- Do a criterion-by-criterion verification; fix gaps or inconsistencies.
- Refine the score and provide tighter justification grounded in the student answer.
- Set routing_decision to 'proceed'.`;
  } else if (stage === "validator") {
    persona = `You are **The Validator**, the quality assurance guardian of the evaluation process.

Refinement loop: ${loops}/2.

Goal: Review the Scrutinizer's refined evaluation for fairness, accuracy, and consistency.

Instructions:
- Audit the pipeline history for fairness, accuracy, and consistency with the rubric.
- Check for missing criteria, unjustified leaps, bias, or unclear feedback.
- If deficiencies remain, set routing_decision to 'revise' (send back for another pass).
- If the evaluation meets a high standard, set routing_decision to 'proceed'.`;
  } else if (stage === "mentor") {
    persona = `You are **The Mentor**, the final agent who turns the validated assessment into a helpful report.

Goal: Produce clear, personalized, actionable feedback for the student.

Instructions:
- Synthesize the final score and reasoning from the pipeline history.
- Write encouraging, specific feedback: celebrate strengths, identify growth areas.
- Provide actionable next steps (concrete improvements / study suggestions).
- Set routing_decision to 'proceed'.`;
  } else {
    persona = `You are an Automated Evaluation Agent.

Goal: Return a reasonable score and clear reasoning.

Instructions:
- Follow the required JSON schema.
- Set routing_decision to 'proceed'.`;
  }

  return `${persona.trim()}

--- PIPELINE HISTORY ---
${String(obs?.pipeline_history ?? "")}

--- DATA ---
Question: ${String(obs?.question ?? "")}
Rubric: ${String(obs?.rubric ?? "")}
Student Answer: ${String(obs?.student_answer ?? "")}
Maximum Score: ${maxScore}

You must output ONLY a raw JSON object. Do not wrap it in markdown formatting (e.g., no \`\`\`json).
{
  "proposed_score": <number strictly between 0 and ${maxScore} inclusive>,
  "agent_reasoning": "<string: your detailed analysis, critique, or feedback for this stage>",
  "routing_decision": "<string: 'proceed' or 'revise'>"
}`;
}

async function completeLlm(prompt) {
  const res = await fetch(LLM_API, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: prompt }],
      temperature: AUTO_TEMPERATURE,
      max_tokens: AUTO_MAX_TOKENS,
    }),
  });
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    // ignore
  }
  if (!res.ok) {
    const detail =
      json && typeof json === "object" && json.detail != null
        ? typeof json.detail === "string"
          ? json.detail
          : JSON.stringify(json.detail)
        : text;
    throw new Error(`LLM HTTP ${res.status}: ${detail}`);
  }
  return String(json?.content ?? "");
}

async function getActionWithRetry(prompt, maxScore) {
  let raw = stripMarkdownJsonFence(await completeLlm(prompt)).trim();
  try {
    return parseAction(raw, maxScore);
  } catch (firstExc) {
    const repair = repairPromptInvalidOutput(raw, maxScore);
    const repaired = stripMarkdownJsonFence(await completeLlm(repair)).trim();
    return parseAction(repaired, maxScore);
  }
}

function initRewardChart() {
  if (typeof Chart === "undefined" || !els.rewardChartCanvas) return;
  rewardChart = new Chart(els.rewardChartCanvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Step reward",
          data: [],
          borderColor: "rgb(99, 102, 241)",
          backgroundColor: "rgba(99, 102, 241, 0.12)",
          fill: true,
          tension: 0.2,
          spanGaps: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Step" },
          ticks: { maxRotation: 45, minRotation: 0 },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Reward" },
        },
      },
    },
  });
}

function clearRewardChart() {
  if (!rewardChart) return;
  rewardChart.data.labels = [];
  rewardChart.data.datasets[0].data = [];
  rewardChart.update();
}

function pushRewardPoint(label, value) {
  if (!rewardChart) return;
  rewardChart.data.labels.push(label);
  rewardChart.data.datasets[0].data.push(Number(value) || 0);
  rewardChart.update("none");
}

function actionLogStr(action) {
  const r = (action.agent_reasoning || "").slice(0, 100) + "...";
  return JSON.stringify({
    proposed_score: action.proposed_score,
    routing: action.routing_decision,
    reasoning: r,
  });
}

async function runEpisodeAuto(taskLabel, taskName, maxSteps, signal) {
  const resetPayload = taskName ? { task_name: taskName } : {};
  const json = await apiPost("/reset", resetPayload);
  let { obs } = extractResult(json);
  renderObs(obs);

  const history = [];
  let lastAction = null;
  let lastReward = 0;

  for (let step = 1; step <= maxSteps; step++) {
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");

    els.autoStatus.textContent = `${taskLabel}: step ${step} / ${maxSteps} (stage ${obs?.current_stage ?? "—"})`;

    const prompt = buildUserPrompt(step, lastAction, lastReward, history, obs);
    let action;
    try {
      action = await getActionWithRetry(prompt, Number(obs?.max_score) || 1);
    } catch (e) {
      history.push(`step=${step} parse_error=${String(e)}`);
      lastAction = null;
      lastReward = 0;
      pushRewardPoint(`${taskLabel}·${step}`, 0);
      continue;
    }

    const stepJson = await apiPost(
      "/step",
      bodyForStep({
        proposed_score: action.proposed_score,
        agent_reasoning: action.agent_reasoning,
        routing_decision: action.routing_decision,
      })
    );
    const { obs: nextObs, reward, done } = extractResult(stepJson);
    const r = Number(reward ?? nextObs?.reward ?? 0);
    pushRewardPoint(`${taskLabel}·${step}`, r);
    lastAction = actionLogStr(action);
    lastReward = r;
    history.push(`step=${step} action=${lastAction} reward=${r.toFixed(2)}`);
    obs = nextObs;
    renderObs(obs);
    if (done || obs?.done) break;
  }
}

async function runAutoRun() {
  clearError();
  const maxSteps = Math.min(
    500,
    Math.max(1, parseInt(els.maxSteps.value, 10) || 10)
  );
  const mode = els.taskSelect.value;
  const tasks =
    mode === "all"
      ? [
          ["easy", "easy"],
          ["medium", "medium"],
          ["hard", "hard"],
        ]
      : [[mode, mode]];

  clearRewardChart();
  autoAbort = new AbortController();
  const signal = autoAbort.signal;
  setAutoRunning(true);

  try {
    for (const [label, name] of tasks) {
      if (signal.aborted) break;
      await runEpisodeAuto(label, name, maxSteps, signal);
    }
    els.autoStatus.textContent = signal.aborted
      ? "Stopped."
      : "Auto-run finished.";
  } catch (e) {
    if (e?.name === "AbortError") {
      els.autoStatus.textContent = "Stopped.";
    } else {
      setError(e);
      els.autoStatus.textContent = "Auto-run failed (see error above).";
    }
  } finally {
    setAutoRunning(false);
    autoAbort = null;
  }
}

async function resetEnv() {
  clearError();
  setLoading(true);
  try {
    const json = await apiPost("/reset", resetBodyForTaskSelect());
    const { obs } = extractResult(json);
    renderObs(obs);
  } catch (e) {
    setError(e);
  } finally {
    setLoading(false);
  }
}

async function stepEnv() {
  clearError();
  setLoading(true);
  try {
    const maxScore = Number(state.obs?.max_score ?? NaN);
    const proposedScoreRaw = els.score.value;
    const proposedScore = proposedScoreRaw === "" ? 0 : Number(proposedScoreRaw);
    const agentReasoning = els.reason.value ?? "";
    const routingDecision = state.routingDecision;

    if (!Number.isFinite(proposedScore)) {
      throw new Error("proposed_score must be a number");
    }
    if (Number.isFinite(maxScore)) {
      const clamped = Math.min(Math.max(proposedScore, 0), maxScore);
      if (clamped !== proposedScore) {
        els.score.value = String(clamped);
      }
    }

    const json = await apiPost(
      "/step",
      bodyForStep({
        proposed_score: Number(els.score.value || 0),
        agent_reasoning: agentReasoning,
        routing_decision: routingDecision,
      })
    );
    const { obs } = extractResult(json);
    renderObs(obs);
  } catch (e) {
    setError(e);
  } finally {
    setLoading(false);
  }
}

function init() {
  els.errorDismiss.addEventListener("click", clearError);

  els.scoreMaxBtn.addEventListener("click", () => {
    const maxScore = Number(state.obs?.max_score ?? NaN);
    if (Number.isFinite(maxScore)) els.score.value = String(maxScore);
  });

  for (const b of els.routeBtns) {
    b.addEventListener("click", () => setRoutingDecision(b.dataset.value));
  }
  setRoutingDecision("proceed");

  els.resetBtn.addEventListener("click", resetEnv);
  els.stepBtn.addEventListener("click", stepEnv);

  els.autoRunBtn.addEventListener("click", () => {
    if (state.autoRunning) return;
    runAutoRun();
  });

  els.autoStopBtn.addEventListener("click", () => {
    if (autoAbort) autoAbort.abort();
  });

  initRewardChart();
  resetEnv();
}

init();
