// Optional dashboard client for repository evidence.
// It fetches `/dashboard/api/metrics`, which reads saved JSON reports produced by
// the pipeline. The charts are therefore a visual inspection aid for the marker,
// not a claim of live production monitoring.

document.addEventListener('DOMContentLoaded', () => {
    // Keep all evidence views on one page so the marker can move between model
    // metrics, SLA, drift, and version comparisons without changing routes.
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabName).classList.add('active');

            if (tabName === 'performance') initPerformanceCharts();
            if (tabName === 'sla') initSlaChart();
        });
    });

    // Initialise the first chart set immediately; other chart sets are built
    // lazily when their tab opens to avoid drawing into hidden canvases.
    initPerformanceCharts();
    updateTimestamp();
});

function initPerformanceCharts() {
    // These values come from version metadata saved by the pipeline. Missing
    // data leaves the dashboard blank rather than inventing a trend.
    fetch('/dashboard/api/metrics')
        .then(r => r.json())
        .then(data => {
            const versions = data.versions || [];
            if (versions.length === 0) return;

            const labels = versions.map(v => v.version);
            const accuracy = versions.map(v => v.accuracy);
            const f1 = versions.map(v => v.f1_weighted);
            const roc = versions.map(v => v.roc_auc);
            const balAcc = versions.map(v => v.balanced_accuracy);

            createChart('accuracyChart', 'Accuracy', labels, accuracy, '#667eea');
            createChart('f1Chart', 'F1 Weighted', labels, f1, '#764ba2');
            createChart('rocChart', 'ROC AUC', labels, roc, '#f093fb');
            createChart('balAccChart', 'Balanced Accuracy', labels, balAcc, '#4facfe');
        })
        .catch(err => console.error('Error loading metrics:', err));
}

function initSlaChart() {
    // SLA charts are based on `scripts/benchmark_api.py` output. They are useful
    // after a Docker or Kind smoke test has produced a benchmark report.
    fetch('/dashboard/api/metrics')
        .then(r => r.json())
        .then(data => {
            const sla = data.sla || {};
            if (!sla.p50_ms) return;

            const ctx = document.getElementById('latencyChart');
            if (!ctx) return;

            // Reopening the tab should replace the old chart instead of stacking
            // Chart.js instances on the same canvas.
            if (window.latencyChartInstance) {
                window.latencyChartInstance.destroy();
            }

            window.latencyChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['P50', 'P95', 'P99', 'Mean'],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [sla.p50_ms, sla.p95_ms, sla.p99_ms, sla.mean_ms],
                        backgroundColor: [
                            '#667eea',
                            '#764ba2',
                            '#f093fb',
                            '#4facfe'
                        ],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'x',
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { callback: v => v + 'ms' }
                        }
                    }
                }
            });
        })
        .catch(err => console.error('Error loading SLA:', err));
}

function createChart(canvasId, label, labels, data, color) {
    // All metric charts use the same scale because these are classification
    // scores between 0 and 1. That makes versions easier to compare visually.
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Rebuild charts safely if the user returns to a tab during the demo.
    if (window[canvasId + '_instance']) {
        window[canvasId + '_instance'].destroy();
    }

    window[canvasId + '_instance'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: color + '15',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: color,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: v => v.toFixed(2)
                    }
                }
            }
        }
    });
}

function updateTimestamp() {
    // This is the browser render time, not the model training time. Training and
    // report timestamps remain in the JSON evidence files.
    const now = new Date().toLocaleString();
    const elem = document.getElementById('update-time');
    if (elem) {
        elem.textContent = now;
    }
}
