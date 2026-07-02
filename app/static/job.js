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
  const jobLayout = document.getElementById("job-layout");
  const textPanel = document.getElementById("text-panel");
  const textPanelMeta = document.getElementById("text-panel-meta");
  const ipaControls = document.getElementById("ipa-controls");
  const ipaVariantSelect = document.getElementById("ipa-variant-select");
  const ipaUnsupportedNote = document.getElementById("ipa-unsupported-note");
  const tabSource = document.getElementById("tab-source");
  const tabTranslation = document.getElementById("tab-translation");
  const sourceTextView = document.getElementById("source-text-view");
  const translationTextView = document.getElementById("translation-text-view");
  const ipaPopover = document.getElementById("ipa-popover");

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
  const pollStartedAt = Date.now();
  const MAX_POLL_MS = 20 * 60 * 1000;
  const MAX_TRANSIENT_404 = 15;
  let transient404Count = 0;
  let textResultLoaded = false;
  let textResultPayload = null;
  let ipaTokens = [];
  let activePopoverToken = null;

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

  function normalizeUnicode(text) {
    return String(text || "").normalize("NFC");
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function clearElement(element) {
    while (element.firstChild) {
      element.removeChild(element.firstChild);
    }
  }

  function appendSourceToken(parent, token) {
    if (token.ipa) {
      const span = document.createElement("span");
      span.className = "ipa-word has-ipa";
      span.tabIndex = 0;
      span.textContent = token.text;
      span.addEventListener("mouseenter", () => {
        showPopover(token, span);
      });
      span.addEventListener("mouseleave", () => {
        if (activePopoverToken === token) {
          hidePopover();
        }
      });
      span.addEventListener("focus", () => {
        showPopover(token, span);
      });
      span.addEventListener("blur", hidePopover);
      parent.appendChild(span);
      return;
    }
    parent.appendChild(document.createTextNode(token.text));
  }

  function hidePopover() {
    activePopoverToken = null;
    ipaPopover.classList.add("hidden");
  }

  function showPopover(token, target) {
    if (!token?.ipa) {
      hidePopover();
      return;
    }
    activePopoverToken = token;
    const matchedLine = token.matched
      ? `<span class="ipa-matched">${escapeHtml(token.text)} → ${escapeHtml(token.matched)}</span><br />`
      : `<strong>${escapeHtml(token.text)}</strong><br />`;
    ipaPopover.innerHTML = `${matchedLine}<span class="ipa-value">${escapeHtml(token.ipa)}</span>`;
    ipaPopover.classList.remove("hidden");

    const rect = target.getBoundingClientRect();
    const popoverRect = ipaPopover.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - popoverRect.width / 2;
    let top = rect.top - popoverRect.height - 8;
    if (top < 8) {
      top = rect.bottom + 8;
    }
    left = Math.max(8, Math.min(left, window.innerWidth - popoverRect.width - 8));
    ipaPopover.style.left = `${left}px`;
    ipaPopover.style.top = `${top}px`;
  }

  function renderPlainTextView(element, text) {
    element.textContent = text || "[No text available]";
  }

  function renderSourceTextView(text) {
    const normalizedText = normalizeUnicode(text);
    clearElement(sourceTextView);

    if (!normalizedText.trim()) {
      sourceTextView.textContent = "[No source text available]";
      return;
    }

    if (!textResultPayload || !textResultPayload.ipa_supported || !ipaTokens.length) {
      sourceTextView.textContent = normalizedText;
      return;
    }

    for (const token of ipaTokens) {
      appendSourceToken(sourceTextView, token);
    }
  }

  function setActiveTab(tabName) {
    const showSource = tabName === "source";
    tabSource.classList.toggle("is-active", showSource);
    tabTranslation.classList.toggle("is-active", !showSource);
    tabSource.setAttribute("aria-selected", showSource ? "true" : "false");
    tabTranslation.setAttribute("aria-selected", showSource ? "false" : "true");
    sourceTextView.classList.toggle("hidden", !showSource);
    translationTextView.classList.toggle("hidden", showSource);
    hidePopover();
  }

  function resolveSourceLangCode(sourceLanguage) {
    const key = normalizeUnicode(sourceLanguage || "")
      .trim()
      .toLowerCase();
    if (key.includes("german") || key === "de" || key === "deutsch") return "de";
    if (key.includes("french") || key === "fr" || key === "français") return "fr";
    if (key.includes("spanish") || key === "es" || key === "español") return "es";
    if (key.includes("italian") || key === "it" || key === "italiano") return "it";
    if (key.includes("english") || key === "en") return "en";
    if (key.includes("japanese") || key === "ja") return "ja";
    if (key.includes("chinese") || key.includes("mandarin") || key === "zh") return "zh";
    return "";
  }

  async function fetchIpaTokens(variant) {
    if (!textResultPayload || !textResultPayload.ipa_supported || !window.jobConfig.jobId) {
      ipaTokens = [];
      return;
    }

    const params = new URLSearchParams();
    if (variant) {
      params.set("variant", variant);
    }
    const query = params.toString();
    const url = `/api/jobs/${window.jobConfig.jobId}/ipa-tokens${query ? `?${query}` : ""}`;
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("Could not load IPA pronunciations.");
    }
    const payload = await response.json();
    ipaTokens = payload.tokens || [];
  }

  function selectedIpaVariant() {
    const variants = textResultPayload?.ipa_variants || [];
    if (variants.length === 1) {
      return variants[0].code;
    }
    if (ipaVariantSelect.value) {
      return ipaVariantSelect.value;
    }
    return textResultPayload?.default_ipa_variant || null;
  }

  function populateVariantSelect() {
    ipaVariantSelect.innerHTML = "";
    const variants = textResultPayload?.ipa_variants || [];
    variants.forEach((variant) => {
      const option = document.createElement("option");
      option.value = variant.code;
      option.textContent = variant.label;
      ipaVariantSelect.appendChild(option);
    });
    if (textResultPayload?.default_ipa_variant) {
      ipaVariantSelect.value = textResultPayload.default_ipa_variant;
    }
  }

  async function renderTextPanel() {
    if (!textResultPayload) return;

    textPanelMeta.textContent = `Source: ${textResultPayload.source_language || "Unknown"} · Target: ${textResultPayload.target_language || "Unknown"}`;

    if (textResultPayload.ipa_supported) {
      const variants = textResultPayload.ipa_variants || [];
      if (variants.length > 1) {
        ipaControls.classList.remove("hidden");
        populateVariantSelect();
      } else {
        ipaControls.classList.add("hidden");
      }
      ipaUnsupportedNote.classList.add("hidden");
      await fetchIpaTokens(selectedIpaVariant());
    } else {
      ipaControls.classList.add("hidden");
      ipaUnsupportedNote.classList.remove("hidden");
      ipaUnsupportedNote.textContent =
        "IPA pronunciation lookup is not available for this source language.";
      ipaTokens = [];
    }

    const sourceLang = resolveSourceLangCode(textResultPayload.source_language);
    if (sourceLang) {
      sourceTextView.lang = sourceLang;
    } else {
      sourceTextView.removeAttribute("lang");
    }

    renderSourceTextView(textResultPayload.full_source_text || "");
    renderPlainTextView(translationTextView, normalizeUnicode(textResultPayload.full_translation || ""));
    setActiveTab("source");
  }

  async function loadTextResult(textResultUrl) {
    if (textResultLoaded || !textResultUrl) return;
    textResultLoaded = true;
    try {
      const response = await fetch(textResultUrl, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error("Could not load text review content.");
      }
      textResultPayload = await response.json();
      jobLayout.classList.add("is-complete");
      textPanel.classList.remove("hidden");
      await renderTextPanel();
    } catch (error) {
      textResultLoaded = false;
      errorArea.textContent = error.message || "Could not load text review content.";
      errorArea.classList.remove("hidden");
    }
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
      loadTextResult(payload.text_result_url);
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
    if (Date.now() - pollStartedAt > MAX_POLL_MS) {
      stopPolling();
      errorArea.textContent =
        "This job appears stalled. Please refresh the page to trigger recovery.";
      errorArea.classList.remove("hidden");
      return;
    }
    try {
      const response = await fetch(window.jobConfig.statusUrl, {
        headers: { Accept: "application/json" },
      });
      if (response.status === 404) {
        transient404Count += 1;
        if (transient404Count <= MAX_TRANSIENT_404) {
          errorArea.classList.add("hidden");
          errorArea.textContent = "";
          return;
        }
      } else {
        transient404Count = 0;
      }
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

  tabSource.addEventListener("click", () => setActiveTab("source"));
  tabTranslation.addEventListener("click", () => setActiveTab("translation"));
  ipaVariantSelect.addEventListener("change", async () => {
    try {
      await fetchIpaTokens(selectedIpaVariant());
      renderSourceTextView(textResultPayload?.full_source_text || "");
    } catch (error) {
      errorArea.textContent = error.message || "Could not load IPA pronunciations.";
      errorArea.classList.remove("hidden");
    }
  });
  window.addEventListener("scroll", hidePopover, true);
  window.addEventListener("resize", hidePopover);

  if (window.jobConfig.initialPayload) {
    renderStatus(window.jobConfig.initialPayload);
  }
  poll();
  timer = setInterval(poll, 1500);
})();
