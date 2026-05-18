(function () {
  // The browser page is a live-demo client for the real Flask API. It uses the
  // feature list injected by Flask so the UI cannot silently drift away from the
  // trained model schema used by `/predict`.
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

  function endpointUrl(configKey, fallbackPath) {
    // Flask supplies route URLs for local, Docker, and Kind runs. The fallback is
    // kept so the page remains understandable if a template value is missing.
    const configured = config[configKey] || fallbackPath;
    return new URL(configured, window.location.origin).toString();
  }

  async function requestJson(url, options) {
    // The live demo should fail with a useful message if the Flask server or
    // Kubernetes port-forward is not running, instead of leaving a silent browser
    // error that is hard for a marker to interpret.
    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error(
        "Prediction API is not reachable. Start the Flask server or restart the Docker/Kind port-forward, then reload this page."
      );
    }

    let data;
    try {
      data = await response.json();
    } catch (error) {
      throw new Error("Prediction API returned a non-JSON response.");
    }

    if (!response.ok) {
      throw new Error(data.error || "The prediction API returned an error.");
    }
    return data;
  }

  function collectPayload() {
    // Collect the exact traffic features expected by the model. Client-side
    // checks help the demo user, while the Flask API still performs the final
    // validation before prediction.
    const payload = {};
    const inputs = [...document.querySelectorAll("[data-feature-input]")];
    for (const input of inputs) {
      const value = input.value.trim();
      if (value === "") {
        throw new Error(`${input.name.replaceAll("_", " ")} is required.`);
      }
      if (input.tagName === "SELECT") {
        payload[input.name] = value;
        continue;
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
    // Show the model's returned label, confidence, and version so the demo proves
    // the browser is calling the trained artefact rather than a static mock.
    const label = data.prediction_label || String(data.prediction);
    predictionValue.textContent = label;
    if (typeof data.confidence === "number") {
      confidenceValue.textContent = `Model confidence: ${(data.confidence * 100).toFixed(1)}%`;
    } else {
      confidenceValue.textContent = "Classification result from the trained model.";
    }
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
      // Check health first because it proves the deployed service has loaded the
      // model before the UI sends a prediction request.
      const health = await requestJson(endpointUrl("healthUrl", "/health"), {method: "GET"});
      if (health.status !== "healthy" || health.model_loaded !== true) {
        throw new Error("Prediction API is not healthy. Check that the model is loaded.");
      }

      const data = await requestJson(endpointUrl("predictUrl", "/predict"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({features: payload}),
      });
      renderPrediction(data);
    } catch (error) {
      showError(error.message || "The prediction request failed.");
    }
  });
})();
