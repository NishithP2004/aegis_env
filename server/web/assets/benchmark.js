const API_BENCH = "/api/benchmark";

const PALETTE = [
  "rgb(99, 102, 241)",
  "rgb(236, 72, 153)",
  "rgb(14, 165, 233)",
  "rgb(34, 197, 94)",
  "rgb(245, 158, 11)",
];

let modelList = [];
let charts = { total: null, steps: null, cumulative: null };

function el(id) {
  return document.getElementById(id);
}

function setError(msg) {
  el("error-text").textContent = msg;
  el("error-banner").classList.remove("hidden");
}

function clearError() {
  el("error-banner").classList.add("hidden");
  el("error-text").textContent = "";
}

function shortName(name, max = 28) {
  if (!name || name.length <= max) return name || "—";
  return name.slice(0, max - 1) + "…";
}

function buildModelSlots() {
  const container = el("model-slots");
  container.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <label class="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Model ${i + 1}</label>
      <select id="bench-model-${i}" class="bench-model-select mt-1 w-full rounded-2xl border border-slate-200 bg-white/80 px-3 py-2 text-xs font-mono shadow-sm outline-none focus:border-indigo-300 focus:ring-4 focus:ring-indigo-200/60">
        <option value="">— Choose —</option>
      </select>
    `;
    container.appendChild(wrap);
  }
}

function fillSelectOptions() {
  for (let i = 0; i < 5; i++) {
    const sel = el(`bench-model-${i}`);
    const v = sel.value;
    sel.innerHTML = '<option value="">— Choose —</option>';
    for (const m of modelList) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    }
    if (modelList.includes(v)) sel.value = v;
  }
}

async function refreshModels() {
  clearError();
  const root = el("api-root").value.trim() || "https://ollama.com/v1";
  el("models-status").textContent = "Loading…";
  try {
    const u = new URL(`${API_BENCH}/models`, window.location.origin);
    u.searchParams.set("api_root", root);
    const res = await fetch(u.toString());
    const text = await res.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      /* ignore */
    }
    if (!res.ok) {
      const detail = json?.detail != null ? (typeof json.detail === "string" ? json.detail : JSON.stringify(json.detail)) : text;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    modelList = Array.isArray(json?.models) ? json.models : [];
    el("models-status").textContent = `${modelList.length} model(s) loaded.`;
    fillSelectOptions();
  } catch (e) {
    el("models-status").textContent = "";
    setError(String(e?.message || e));
  }
}

function gatherSelectedModels() {
  const out = [];
  for (let i = 0; i < 5; i++) {
    const v = el(`bench-model-${i}`).value.trim();
    if (v) out.push(v);
  }
  return out;
}

function destroyChart(c) {
  if (c) {
    c.destroy();
  }
}

function renderCharts(payload) {
  const results = payload?.results || [];

  destroyChart(charts.total);
  destroyChart(charts.steps);
  destroyChart(charts.cumulative);

  const labels = results.map((r) => shortName(r.model, 32));
  const totals = results.map((r) => Number(r.total_reward) || 0);
  const steps = results.map((r) => Number(r.steps) || 0);

  const ctxT = el("chart-total");
  const ctxS = el("chart-steps");
  const ctxC = el("chart-cumulative");

  charts.total = new Chart(ctxT, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Total reward",
          data: totals,
          backgroundColor: PALETTE.map((c) => c.replace("rgb", "rgba").replace(")", ", 0.55)")),
          borderColor: PALETTE,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: { beginAtZero: true, title: { display: true, text: "Sum of step rewards" } },
      },
      plugins: { legend: { display: false } },
    },
  });

  charts.steps = new Chart(ctxS, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Steps",
          data: steps,
          backgroundColor: PALETTE.map((c) => c.replace("rgb", "rgba").replace(")", ", 0.45)")),
          borderColor: PALETTE,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: { beginAtZero: true, title: { display: true, text: "Environment steps taken" } },
      },
      plugins: { legend: { display: false } },
    },
  });

  let maxLen = 0;
  for (const r of results) {
    const arr = r.rewards || [];
    if (arr.length > maxLen) maxLen = arr.length;
  }
  const stepLabels = Array.from({ length: maxLen }, (_, i) => String(i + 1));

  const cumDatasets = results.map((r, idx) => {
    const rewards = r.rewards || [];
    let run = 0;
    const cum = rewards.map((x) => {
      run += Number(x) || 0;
      return run;
    });
    while (cum.length < maxLen) cum.push(null);
    return {
      label: shortName(r.model, 24),
      data: cum,
      borderColor: PALETTE[idx % PALETTE.length],
      backgroundColor: "transparent",
      tension: 0.15,
      spanGaps: false,
    };
  });

  charts.cumulative = new Chart(ctxC, {
    type: "line",
    data: {
      labels: stepLabels,
      datasets: cumDatasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Step" } },
        y: { beginAtZero: true, title: { display: true, text: "Cumulative reward" } },
      },
    },
  });
}

function renderTable(payload) {
  const tbody = el("bench-table-body");
  tbody.innerHTML = "";
  for (const r of payload?.results || []) {
    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-100";
    tr.innerHTML = `
      <td class="py-2 pr-3 font-mono text-[11px] text-slate-800">${escapeHtml(r.model)}</td>
      <td class="py-2 pr-3 font-mono">${(Number(r.total_reward) || 0).toFixed(4)}</td>
      <td class="py-2 pr-3">${r.steps ?? "—"}</td>
      <td class="py-2 text-rose-700">${r.error ? escapeHtml(r.error) : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function runBenchmark() {
  clearError();
  const models = gatherSelectedModels();
  if (models.length !== 5) {
    setError("Select exactly five distinct models (one per slot).");
    return;
  }
  const uniq = new Set(models);
  if (uniq.size !== 5) {
    setError("Each of the five slots must use a different model.");
    return;
  }

  const task = el("bench-task").value;
  const maxSteps = Math.min(200, Math.max(1, parseInt(el("bench-max-steps").value, 10) || 10));
  const seedRaw = el("bench-seed").value.trim();
  const seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
  if (seedRaw !== "" && !Number.isFinite(seed)) {
    setError("Seed must be a valid integer.");
    return;
  }

  const apiBase = el("api-root").value.trim() || "https://ollama.com/v1";
  const apiKey = el("api-key").value.trim();

  el("btn-run-benchmark").disabled = true;
  el("bench-status").textContent = "Running five episodes (this can take a while)…";

  try {
    const body = {
      models,
      task_name: task,
      max_steps: maxSteps,
      seed,
      api_base_url: apiBase,
    };
    if (apiKey) body.api_key = apiKey;

    const res = await fetch(`${API_BENCH}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      /* ignore */
    }
    if (!res.ok) {
      const detail = json?.detail != null ? (typeof json.detail === "string" ? json.detail : JSON.stringify(json.detail)) : text;
      throw new Error(detail || `HTTP ${res.status}`);
    }

    renderTable(json);
    renderCharts(json);
    el("bench-status").textContent = "Done.";
  } catch (e) {
    setError(String(e?.message || e));
    el("bench-status").textContent = "";
  } finally {
    el("btn-run-benchmark").disabled = false;
  }
}

function init() {
  buildModelSlots();
  el("error-dismiss").addEventListener("click", clearError);
  el("btn-refresh-models").addEventListener("click", refreshModels);
  el("btn-run-benchmark").addEventListener("click", runBenchmark);
  refreshModels();
}

init();
