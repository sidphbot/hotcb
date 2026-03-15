/**
 * hotcb dashboard — initialization and data loading
 */

function dismissChartWaiting() {
  var el = document.getElementById('chartWaiting');
  if (el && !el.classList.contains('hidden')) {
    el.classList.add('hidden');
    setTimeout(function() { el.style.display = 'none'; }, 500);
  }
}

async function initialLoad() {
  // Fetch centralized config before other init
  var cfg = await api('GET', '/api/config');
  if (cfg) {
    S.config = cfg;
    // Build dynamic controls from config
    if (cfg.controls && cfg.controls.length) {
      buildControls(cfg.controls);
    }
  }

  // Status
  var status = await api('GET', '/api/status');
  if (status) {
    if (status.freeze) $('#freezeSelect').value = status.freeze.mode || 'off';
    if (status.run_dir) S.runs = [status.run_dir];
  }

  // Metric history — load full run for external projects (LTTB handles rendering)
  var hist = await api('GET', '/api/metrics/history?last_n=50000');
  if (hist && hist.records && hist.records.length > 0) {
    dismissChartWaiting();
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
    // Restore pinned metrics now that we have data
    if (S._pendingPinnedMetrics) {
      S._pendingPinnedMetrics.forEach(function(name) {
        if (S.metricNames.has(name)) {
          S.pinnedMetrics.add(name);
          createMetricCard(name);
        }
      });
      delete S._pendingPinnedMetrics;
    }
    updateMetricToggles();
    updateChart();
    computeHealth();
  } else {
    // No data — clear any pending pinned metrics
    delete S._pendingPinnedMetrics;
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
  if (ap && ap.mode) {
    $('#autopilotMode').value = ap.mode;
    _updateAIConfigVisibility(ap.mode);
  }
  pollAutopilotStatus();

  // Restore controls from server state (overrides stale localStorage)
  var ctrlState = await api('GET', '/api/state/controls');
  if (ctrlState) {
    // Build/rebuild controls from live MutableState data
    if (ctrlState.controls && ctrlState.controls.length) {
      buildControls(ctrlState.controls);
      if (S.config) S.config.controls = ctrlState.controls;
    }

    // Sync sliders from latest metrics using dynamic sync
    var m = ctrlState.latest_metrics || {};
    if (Object.keys(m).length > 0) {
      syncSlidersFromApplied(m);
    }
    // Sync from last applied opt params as fallback
    var op = ctrlState.last_opt_params || {};
    if (Object.keys(op).length > 0) {
      syncSlidersFromApplied(op);
    }
    if (ctrlState.last_loss_params && Object.keys(ctrlState.last_loss_params).length > 0) {
      syncSlidersFromApplied(ctrlState.last_loss_params);
    }

    // Sync training config
    var rc = ctrlState.run_config || {};
    if (rc.config_id) {
      var sel = document.getElementById('trainConfig');
      if (sel) {
        sel.value = rc.config_id;
        _updateConfigControls(rc.config_id);
      }
    }
    if (rc.max_steps) {
      var msEl = document.getElementById('trainMaxSteps');
      if (msEl) msEl.value = rc.max_steps;
    }
    if (rc.step_delay !== undefined) {
      var sdEl = document.getElementById('trainStepDelay');
      if (sdEl) sdEl.value = rc.step_delay;
    }
    if (rc.seed !== undefined && rc.seed !== null) {
      var seedEl = document.getElementById('trainSeed');
      if (seedEl) seedEl.value = rc.seed;
    }
    // Step counter
    if (ctrlState.latest_step) {
      var stepEl = document.getElementById('stepValue');
      if (stepEl) stepEl.textContent = ctrlState.latest_step;
    }
  }

  // Load training capabilities (informational)
  loadCapabilities();

  // Periodic updates
  startForecastPolling();
  var _alertPollMs = (S.config && S.config.ui) ? S.config.ui.alert_poll_interval : 15000;
  S._alertInterval = setInterval(fetchAlerts, _alertPollMs);

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
  // Health card collapse toggle
  var healthToggle = document.getElementById('healthToggle');
  var healthDetails = document.getElementById('healthDetails');
  if (healthToggle && healthDetails) {
    // Start collapsed
    healthDetails.classList.add('collapsed');
    healthToggle.classList.add('collapsed');
    healthToggle.addEventListener('click', function() {
      healthDetails.classList.toggle('collapsed');
      healthToggle.classList.toggle('collapsed');
    });
  }
  $('#btnTour').addEventListener('click', startTour);
  initRecipeEditor();
  initAutopilotRulesEditor();
  initChat();
  initNotifications();
  initCallHelp();
  initSaveAsRecipe();
  initCallbackDiagnostics();
  initConfigWizard();
  initCompare();
  createMetricsChart();
  initStepRangeControls();
  // Normalize toggle
  var normBtn = document.getElementById('btnNormalize');
  if (normBtn) {
    normBtn.addEventListener('click', function() {
      _chartNormalize = !_chartNormalize;
      normBtn.classList.toggle('btn-accent', _chartNormalize);
      updateChart();
    });
  }
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
    // Defer pinned metrics restoration until data is loaded
    if (savedState.pinnedMetrics && savedState.pinnedMetrics.length) {
      S._pendingPinnedMetrics = savedState.pinnedMetrics;
    }
    // Knob state is now handled by dynamic controls from server
  }

  // Persist UI state periodically and before page unload
  var _stateSaveMs = (S.config && S.config.ui) ? S.config.ui.state_save_interval : 5000;
  S._saveStateInterval = setInterval(saveUIState, _stateSaveMs);
  window.addEventListener('beforeunload', saveUIState);
})();
