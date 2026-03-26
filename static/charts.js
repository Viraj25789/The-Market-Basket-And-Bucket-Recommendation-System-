/**
 * charts.js
 * ---------
 * Initializes both Chart.js charts on the dashboard.
 * Called once from index.html after the DOM is ready.
 */

/**
 * Render the item-frequency bar chart.
 * @param {string[]} labels  - Item names
 * @param {number[]} counts  - Purchase counts
 */
function renderItemChart(labels, counts) {
  const ctx = document.getElementById('itemChart');
  if (!ctx) return;

  new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Purchases',
        data: counts,
        backgroundColor: 'rgba(0, 229, 160, 0.65)',
        borderColor:     'rgba(0, 229, 160, 0.9)',
        borderWidth: 1,
        borderRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#6b7a99', maxRotation: 35, font: { size: 11 } },
          grid:  { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#6b7a99' },
          grid:  { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}

/**
 * Render the association-rules combo chart (bars = confidence, line = lift).
 * @param {string[]} labels     - Rule labels like "Milk→Bread"
 * @param {number[]} confidence - Confidence values
 * @param {number[]} lift       - Lift values
 */
function renderRulesChart(labels, confidence, lift) {
  const ctx = document.getElementById('rulesChart');
  if (!ctx) return;

  new Chart(ctx.getContext('2d'), {
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'Confidence',
          data: confidence,
          backgroundColor: 'rgba(0, 229, 160, 0.35)',
          borderColor:     'rgba(0, 229, 160, 0.7)',
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Lift',
          data: lift,
          borderColor:  '#f5a623',
          borderWidth:  2.5,
          pointBackgroundColor: '#f5a623',
          tension: 0.4,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#8a97b8', boxWidth: 12, font: { size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: '#6b7a99', font: { size: 10 } },
          grid:  { display: false },
        },
        y: {
          position: 'left',
          ticks: { color: '#6b7a99' },
          grid:  { color: 'rgba(255,255,255,0.05)' },
        },
        y1: {
          position: 'right',
          ticks: { color: '#f5a623' },
          grid:  { display: false },
        },
      },
    },
  });
}