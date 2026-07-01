(function () {
  if (!window.jobConfig || !window.jobConfig.statusUrl) return;

  const statusMessage = document.getElementById("status-message");
  const progressBar = document.getElementById("progress-bar");
  const progressPercent = document.getElementById("progress-percent");
  const batchMessage = document.getElementById("batch-message");
  const stageSummary = document.getElementById("stage-summary");
  const stageItems = Array.from(document.querySelectorAll(".stage-item"));
  const resultArea = document.getElementById("result-area");
  const downloadLink = document.getElementById("download-link");
  const downloadLogLink = document.getElementById("download-log-link");
  const errorArea = document.getElementById("error-area");

  const STAGES = [
    { id: "queued", label: "Queued", statuses: ["queued"] },
    {
      id: "extracting",
      label: "Preparing score pages",
      statuses: ["validating", "rendering"],
    },
    { id: "translating", label: "Translating text", statuses: ["analysing"] },
    {
      id: "rebuilding",
      label: "Building translated PDF",
      statuses: ["creating_pdf"],
    },
    { id: "ready", label: "Ready to download", statuses: ["complete"] },
  ];

  let timer = null;
  let lastStageIndex = 0;

  function stopPolling() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  function setProgress(progress) {
    const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
    progressBar.style.width = `${pct}%`;
    progressPercent.textContent = `${pct}%`;
  }

  function resolveStageIndex(payload) {
    const status = payload.status || "";
    const directMatch = STAGES.findIndex((stage) => stage.statuses.includes(status));
    if (directMatch >= 0) return directMatch;

    const progress = typeof payload.progress === "number" ? payload.progress : 0;
    if (progress >= 0.9) return 3;
    if (progress >= 0.25) return 2;
    if (progress >= 0.02) return 1;
    return 0;
  }

  function setStageState(payload) {
    if (!stageItems.length || !stageSummary) return;

    const failed = payload.status === "failed";
    const stageIndex = failed ? lastStageIndex : resolveStageIndex(payload);
    lastStageIndex = stageIndex;

    stageItems.forEach((item, index) => {
      item.classList.remove("is-upcoming", "is-active", "is-complete", "is-failed");
      if (index < stageIndex) {
        item.classList.add("is-complete");
      } else if (index === stageIndex) {
        item.classList.add(failed ? "is-failed" : "is-active");
      } else {
        item.classList.add("is-upcoming");
      }
    });

    if (failed) {
      stageSummary.textContent = `Stopped at step ${stageIndex + 1} of ${
        STAGES.length
      }: ${STAGES[stageIndex].label}`;
      return;
    }

    stageSummary.textContent = `Step ${stageIndex + 1} of ${STAGES.length}: ${
      STAGES[stageIndex].label
    }`;
  }

  function renderStatus(payload) {
    statusMessage.textContent = payload.message || "Processing...";
    setProgress(payload.progress || 0);
    setStageState(payload);

    if (payload.current_batch && payload.total_batches) {
      batchMessage.textContent = `Currently translating batch ${payload.current_batch} of ${payload.total_batches}`;
    } else {
      batchMessage.textContent = "";
    }

    if (payload.status === "complete" && payload.download_url) {
      resultArea.classList.remove("hidden");
      downloadLink.setAttribute("href", payload.download_url);
      if (downloadLogLink) {
        if (payload.log_url) {
          downloadLogLink.classList.remove("hidden");
          downloadLogLink.setAttribute("href", payload.log_url);
        } else {
          downloadLogLink.classList.add("hidden");
        }
      }
      errorArea.classList.add("hidden");
      stopPolling();
    } else if (payload.status === "failed") {
      const message = payload.error || "Processing failed.";
      errorArea.textContent = message;
      errorArea.classList.remove("hidden");
      stopPolling();
    } else {
      errorArea.classList.add("hidden");
      errorArea.textContent = "";
    }
  }

  async function poll() {
    try {
      const response = await fetch(window.jobConfig.statusUrl, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error("Could not load job status.");
      }
      const payload = await response.json();
      renderStatus(payload);
    } catch (error) {
      errorArea.textContent = error.message || "Could not load job status.";
      errorArea.classList.remove("hidden");
    }
  }

  if (window.jobConfig.initialPayload) {
    renderStatus(window.jobConfig.initialPayload);
  }
  poll();
  timer = setInterval(poll, 1500);
})();

