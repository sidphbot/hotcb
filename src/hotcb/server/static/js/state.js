/**
 * hotcb dashboard — global state
 */

const COLORS = ['#00d4aa','#3d9eff','#ff9833','#ff4d5e','#9966ff','#33dd77','#ffd233','#33ccdd','#ff66aa','#aabb44'];

const S = {
  ws: null,
  metricsData: {},       // {metricName: [{step, value}]}
  appliedData: [],
  metricNames: new Set(),
  metricColors: {},
  colorIdx: 0,
  latestMetrics: {},
  chartInstance: null,
  pinnedMetrics: new Set(),
  metricCardCharts: {},
  focusMetric: null,
  manifoldCtx: null,
  featureCtx: null,
  recipeEntries: [],
  chatHistory: [],
  alerts: [],
  runs: [],
  activeRuns: new Set([0]),
  healthScore: 0,
};

function getColor(name) {
  if (!S.metricColors[name]) {
    S.metricColors[name] = COLORS[S.colorIdx % COLORS.length];
    S.colorIdx++;
  }
  return S.metricColors[name];
}

function saveUIState() {
  var state = {
    activeTab: null,
    trainConfig: null,
    pinnedMetrics: [],
    knobs: {},
    metricVisibility: {},
    focusMetric: null,
  };
  // Active tab
  var activeTab = document.querySelector('.tabs .tab.active[data-tab]');
  if (activeTab) state.activeTab = activeTab.dataset.tab;
  // Train config
  var sel = document.getElementById('trainConfig');
  if (sel) state.trainConfig = sel.value;
  // Pinned metrics — only save if there's data (avoids persisting stale pins)
  if (S.metricNames && S.metricNames.size > 0) {
    state.pinnedMetrics = Array.from(S.pinnedMetrics || []);
  }
  // Don't persist focusMetric — it should reset with the session
  // Knobs
  var lr = document.getElementById('knobLr');
  var wd = document.getElementById('knobWd');
  if (lr) state.knobs.lr = lr.value;
  if (wd) state.knobs.wd = wd.value;
  // Metric visibility
  if (S.metricNames) {
    S.metricNames.forEach(function(name) {
      state.metricVisibility[name] = S.metricsData[name] ? true : false;
    });
  }
  localStorage.setItem('hotcb-ui-state', JSON.stringify(state));
}

function loadUIState() {
  try {
    var raw = localStorage.getItem('hotcb-ui-state');
    if (!raw) return null;
    return JSON.parse(raw);
  } catch(e) { return null; }
}
