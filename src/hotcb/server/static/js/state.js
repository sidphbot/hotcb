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
