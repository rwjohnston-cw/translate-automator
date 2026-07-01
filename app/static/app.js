(function () {
  const form = document.getElementById("upload-form");
  if (!form) return;

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("pdf-file");
  const pickFileBtn = document.getElementById("pick-file-btn");
  const selectedFile = document.getElementById("selected-file");
  const languageSelect = document.getElementById("target-language");
  const customWrapper = document.getElementById("custom-language-wrapper");
  const customInput = document.getElementById("custom-target-language");
  const testingModeInput = document.getElementById("testing-mode");
  const testingPanel = document.getElementById("testing-panel");
  const providerSelect = document.getElementById("llm-provider");
  const modelSelect = document.getElementById("llm-model-select");
  const modelCustom = document.getElementById("llm-model-custom");
  const reasoningSelect = document.getElementById("llm-reasoning-effort");
  const reasoningHelp = document.getElementById("llm-reasoning-help");
  const positioningVariantSelect = document.getElementById("positioning-variant");
  const translationWorkflowSelect = document.getElementById("translation-workflow");
  const ownedBatchSizeInput = document.getElementById("owned-batch-size");
  const contextPagesInput = document.getElementById("context-pages");
  const formError = document.getElementById("form-error");
  const submitBtn = document.getElementById("submit-btn");
  const modelOptionsData = document.getElementById("model-options-data");
  const reasoningOptionsData = document.getElementById("reasoning-options-data");
  const reasoningApiFieldData = document.getElementById("reasoning-api-field-data");
  const modelOptions = modelOptionsData ? JSON.parse(modelOptionsData.textContent || "{}") : {};
  const reasoningOptions = reasoningOptionsData
    ? JSON.parse(reasoningOptionsData.textContent || "{}")
    : {};
  const reasoningApiField = reasoningApiFieldData
    ? JSON.parse(reasoningApiFieldData.textContent || "{}")
    : {};
  const maxUploadMb = Number.parseFloat(form.dataset.maxUploadMb || "");
  const maxUploadMbLabel = Number.isFinite(maxUploadMb)
    ? String(maxUploadMb).replace(/\.0+$/, "")
    : "4.5";

  function showError(message) {
    formError.textContent = message || "";
  }

  function updateFileLabel() {
    const file = fileInput.files && fileInput.files[0];
    selectedFile.textContent = file ? file.name : "No file selected";
  }

  function updateCustomLanguageVisibility() {
    const isOther = languageSelect.value === "Other...";
    customWrapper.classList.toggle("hidden", !isOther);
    customInput.required = isOther;
    if (!isOther) customInput.value = "";
  }

  function updateTestingVisibility() {
    const isTesting = !!testingModeInput.checked;
    testingPanel.classList.toggle("hidden", !isTesting);
    providerSelect.disabled = !isTesting;
    modelSelect.disabled = !isTesting;
    modelCustom.disabled = !isTesting;
    reasoningSelect.disabled = !isTesting;
    positioningVariantSelect.disabled = !isTesting;
    if (translationWorkflowSelect) {
      translationWorkflowSelect.disabled = !isTesting;
    }
    ownedBatchSizeInput.disabled = !isTesting;
    contextPagesInput.disabled = !isTesting;
  }

  function updateModelOptions() {
    const provider = providerSelect.value;
    const options = modelOptions[provider] || [];
    modelSelect.innerHTML = "";
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Use server default";
    modelSelect.appendChild(defaultOption);

    options.forEach(function (model) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    });
    if (options.length) {
      modelSelect.value = options[0];
      modelCustom.value = options[0];
    } else {
      modelSelect.value = "";
      modelCustom.value = "";
    }
    updateReasoningOptions();
  }

  function updateReasoningOptions() {
    const provider = providerSelect.value;
    const model = (modelCustom.value || modelSelect.value || "").trim();
    const providerOptions = reasoningOptions[provider] || {};
    const options = providerOptions[model] || providerOptions.__default__ || [];
    const previousValue = reasoningSelect.value;

    reasoningSelect.innerHTML = "";
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Use provider default";
    reasoningSelect.appendChild(defaultOption);

    options.forEach(function (effort) {
      const option = document.createElement("option");
      option.value = effort;
      option.textContent = effort;
      reasoningSelect.appendChild(option);
    });

    if (options.includes(previousValue)) {
      reasoningSelect.value = previousValue;
    } else {
      reasoningSelect.value = "";
    }

    let apiField = reasoningApiField[provider];
    if (provider === "gemini" && model.startsWith("gemini-2.5-")) {
      apiField = "generationConfig.thinkingConfig.thinkingBudget";
    }
    if (reasoningHelp) {
      if (apiField && options.length) {
        reasoningHelp.textContent =
          "Sent as " + apiField + ". Supported values for this model: " + options.join(", ") + ".";
      } else if (apiField) {
        reasoningHelp.textContent = "Sent as " + apiField + ".";
      } else {
        reasoningHelp.textContent = "";
      }
    }
  }

  function assignDroppedFile(fileList) {
    if (!fileList || !fileList.length) return;
    const dt = new DataTransfer();
    dt.items.add(fileList[0]);
    fileInput.files = dt.files;
    updateFileLabel();
  }

  pickFileBtn.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", updateFileLabel);
  languageSelect.addEventListener("change", updateCustomLanguageVisibility);
  testingModeInput.addEventListener("change", updateTestingVisibility);
  providerSelect.addEventListener("change", function () {
    updateModelOptions();
  });
  modelSelect.addEventListener("change", function () {
    if (modelSelect.value) {
      modelCustom.value = modelSelect.value;
    } else {
      modelCustom.value = "";
    }
    updateReasoningOptions();
  });
  modelCustom.addEventListener("input", updateReasoningOptions);

  dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    assignDroppedFile(event.dataTransfer.files);
  });

  dropZone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    showError("");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showError("Please select a PDF file.");
      return;
    }
    if (!languageSelect.value) {
      showError("Please choose a target language.");
      return;
    }

    const formData = new FormData();
    formData.append("pdf_file", file);
    formData.append("target_language", languageSelect.value);
    if (languageSelect.value === "Other...") {
      formData.append("custom_target_language", customInput.value.trim());
    }
    if (testingModeInput.checked) {
      formData.append("testing_mode", "on");
      formData.append("llm_provider", providerSelect.value);
      formData.append("llm_model", modelCustom.value.trim());
      formData.append("llm_reasoning_effort", reasoningSelect.value);
      formData.append("positioning_variant", positioningVariantSelect.value);
      if (translationWorkflowSelect) {
        formData.append("translation_workflow", translationWorkflowSelect.value);
      }
      formData.append("owned_batch_size", ownedBatchSizeInput.value.trim());
      formData.append("context_pages", contextPagesInput.value.trim());
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Uploading...";
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });
      let payload = null;
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        payload = await response.json();
      }
      if (!response.ok) {
        if (response.status === 413) {
          throw new Error(
            `The uploaded file is too large. Maximum upload size is ${maxUploadMbLabel} MB.`
          );
        }
        throw new Error((payload && payload.detail) || "Upload failed.");
      }
      if (!payload || !payload.page_url) {
        throw new Error("Upload completed but no job page was returned.");
      }
      window.location.assign(payload.page_url);
    } catch (error) {
      showError(error.message || "Upload failed.");
      submitBtn.disabled = false;
      submitBtn.textContent = "Translate score";
    }
  });

  updateFileLabel();
  updateCustomLanguageVisibility();
  updateModelOptions();
  updateReasoningOptions();
  updateTestingVisibility();
})();

