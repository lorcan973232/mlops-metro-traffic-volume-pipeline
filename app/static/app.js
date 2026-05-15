(function () {
  const config = window.MODEL_UI_CONFIG || {};
  const form = document.querySelector("#prediction-form");
  const useExampleButton = document.querySelector("#use-example");
  const resultPanel = document.querySelector("#result-panel");
  const errorPanel = document.querySelector("#error-panel");
  const predictionValue = document.querySelector("#prediction-value");
  const confidenceValue = document.querySelector("#confidence-value");
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

  function renderPrediction(data) {
    const unit = data.unit || config.targetUnit || "load units";
    predictionValue.textContent = `${Number(data.prediction).toFixed(2)} ${unit}`;
    confidenceValue.textContent = "Regression estimate from the trained model.";
    resultModelVersion.textContent = `Model version: ${data.model_version || config.modelVersion}`;
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
