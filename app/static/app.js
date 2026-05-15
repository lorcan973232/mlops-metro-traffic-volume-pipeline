(function () {
  const config = window.MODEL_UI_CONFIG || {};
  const form = document.querySelector("#prediction-form");
  const useExampleButton = document.querySelector("#use-example");
  const resultPanel = document.querySelector("#result-panel");
  const errorPanel = document.querySelector("#error-panel");
  const predictionValue = document.querySelector("#prediction-value");
  const confidenceValue = document.querySelector("#confidence-value");
  const probabilityList = document.querySelector("#probability-list");
  const resultModelVersion = document.querySelector("#result-model-version");
  const errorMessage = document.querySelector("#error-message");

  function setPanelVisibility(resultVisible, errorVisible) {
    resultPanel.hidden = !resultVisible;
    errorPanel.hidden = !errorVisible;
  }

  function showError(message) {
    errorMessage.textContent = message;
    setPanelVisibility(false, true);
  }

  function collectPayload() {
    const payload = {};
    const inputs = [...document.querySelectorAll("[data-feature-input]")];
    for (const input of inputs) {
      const value = input.value.trim();
      if (value === "") {
        throw new Error(`${input.name.replaceAll("_", " ")} is required.`);
      }
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue)) {
        throw new Error(`${input.name.replaceAll("_", " ")} must be numeric.`);
      }
      payload[input.name] = numericValue;
    }

    const expected = config.expectedFeatures || [];
    const missing = expected.filter((feature) => !(feature in payload));
    if (missing.length > 0) {
      throw new Error(`Missing required model features: ${missing.join(", ")}`);
    }
    return payload;
  }

  function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(2)}%`;
  }

  function renderProbabilities(probabilities) {
    probabilityList.innerHTML = "";
    for (const [label, probability] of Object.entries(probabilities || {})) {
      const row = document.createElement("div");
      row.className = "probability-row";

      const labelElement = document.createElement("span");
      labelElement.textContent = label;

      const track = document.createElement("div");
      track.className = "probability-track";

      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = formatPercent(probability);
      track.appendChild(fill);

      const valueElement = document.createElement("span");
      valueElement.textContent = formatPercent(probability);

      row.append(labelElement, track, valueElement);
      probabilityList.appendChild(row);
    }
  }

  function renderPrediction(data) {
    const probabilities = data.probabilities || {};
    const confidence = probabilities[data.prediction];
    predictionValue.textContent = data.prediction || "Unknown";
    confidenceValue.textContent = Number.isFinite(confidence)
      ? `Confidence: ${formatPercent(confidence)}`
      : "Confidence unavailable";
    resultModelVersion.textContent = `Model version: ${data.model_version || config.modelVersion}`;
    renderProbabilities(probabilities);
    setPanelVisibility(true, false);
  }

  useExampleButton.addEventListener("click", () => {
    const example = config.examplePayload || {};
    for (const input of document.querySelectorAll("[data-feature-input]")) {
      input.value = example[input.name] ?? input.dataset.exampleValue ?? "";
    }
    setPanelVisibility(false, false);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setPanelVisibility(false, false);

    let payload;
    try {
      payload = collectPayload();
    } catch (error) {
      showError(error.message);
      return;
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({features: payload}),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "The prediction API returned an error.");
      }
      renderPrediction(data);
    } catch (error) {
      showError(error.message || "The prediction request failed.");
    }
  });
})();
