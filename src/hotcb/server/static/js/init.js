/**
 * hotcb dashboard — initialization and data loading
 */

async function initialLoad() {
  // Status
  var status = await api('GET', '/api/status');
  if (status) {
    if (status.freeze) $('#freezeSelect').value = status.freeze.mode || 'off';
    if (status.run_dir) S.runs = [status.run_dir];
  }

  // Metric history
  var hist = await api('GET', '/api/metrics/history?last_n=2000');
  if (hist && hist.records) {
    hist.records.forEach(function(rec) {
      var step = rec.step || 0;
      var metrics = rec.metrics || {};
      Object.assign(S.latestMetrics, metrics);
      Object.keys(metrics).forEach(function(name) {
        var value = metrics[name];
        if (typeof value !== 'number') return;
        S.metricNames.add(name);
        if (!S.metricsData[name]) S.metricsData[name] = [];
        S.metricsData[name].push({step: step, value: value});
      });
    });
    updateMetricToggles();
    updateChart();
    computeHealth();
  }

  // Applied history
  var applied = await api('GET', '/api/applied/history?last_n=200');
  if (applied && applied.records) {
    applied.records.forEach(function(rec) {
      S.appliedData.push(rec);
      addTimelineItem(rec);
    });
  }

  // Autopilot mode
  var ap = await api('GET', '/api/autopilot/mode');
  if (ap && ap.mode) $('#autopilotMode').value = ap.mode;
  pollAutopilotStatus();

  // Periodic updates
  startForecastPolling();
  setInterval(fetchAlerts, 15000);

  // Show tour for first-time users (with delay to let UI settle)
  if (shouldShowTour()) {
    setTimeout(startTour, 2000);
  }
}

/* ================================================================ */
/* Bootstrap                                                         */
/* ================================================================ */
(function() {
  // Restore theme
  var savedTheme = localStorage.getItem('hotcb-theme') || 'midnight';
  document.documentElement.setAttribute('data-theme', savedTheme);
  var themeSel = document.querySelector('#themeSelect');
  if (themeSel) themeSel.value = savedTheme;

  initTabs();
  initControls();
  $('#btnTour').addEventListener('click', startTour);
  initRecipeEditor();
  initChat();
  initNotifications();
  initCallHelp();
  initSaveAsRecipe();
  initCallbackDiagnostics();
  initConfigWizard();
  initCompare();
  createMetricsChart();
  initialLoad();
  connectWS();

  // Restore UI state from localStorage
  var savedState = loadUIState();
  if (savedState) {
    if (savedState.activeTab) {
      var tab = document.querySelector('.tab[data-tab="' + savedState.activeTab + '"]');
      if (tab) tab.click();
    }
    if (savedState.trainConfig) {
      var sel = document.getElementById('trainConfig');
      if (sel) {
        sel.value = savedState.trainConfig;
        sel.dispatchEvent(new Event('change'));
      }
    }
    if (savedState.pinnedMetrics && savedState.pinnedMetrics.length) {
      savedState.pinnedMetrics.forEach(function(name) {
        S.pinnedMetrics.add(name);
      });
    }
    if (savedState.knobs) {
      if (savedState.knobs.lr) {
        var lr = document.getElementById('knobLr');
        if (lr) { lr.value = savedState.knobs.lr; lr.dispatchEvent(new Event('input')); }
      }
      if (savedState.knobs.wd) {
        var wd = document.getElementById('knobWd');
        if (wd) { wd.value = savedState.knobs.wd; wd.dispatchEvent(new Event('input')); }
      }
    }
  }

  // Persist UI state periodically and before page unload
  setInterval(saveUIState, 5000);
  window.addEventListener('beforeunload', saveUIState);
})();
