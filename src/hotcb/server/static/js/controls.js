/**
 * hotcb dashboard — knobs, freeze, autopilot
 *
 * Controls are generated dynamically from /api/config controls field
 * (populated by MutableState.describe_all() on the server).
 */

function lrFromSlider(v) { return Math.pow(10, parseFloat(v)); }
function sliderFromLr(lr) { return Math.log10(lr); }

var _applyDebounceTimer = null;
var _applyQueue = [];
var _lastApplyStep = 0;
var _healthEma = null;  // EMA-smoothed health score

// Track last-applied slider values for staged-change highlighting
var _appliedKnobs = {};

/* ================================================================ */
/* Dynamic control generation from actuator metadata                 */
/* ================================================================ */

function buildControls(controlSpecs) {
  var panel = document.getElementById('knobPanel');
  if (!panel) return;
  if (!controlSpecs || !controlSpecs.length) {
    panel.innerHTML = '<div style="font-size:10px;color:var(--text-muted)">No controls available</div>';
    return;
  }
  panel.innerHTML = '';

  // Group by group field
  var groups = {};
  var groupOrder = [];
  controlSpecs.forEach(function(spec) {
    var g = spec.group || 'other';
    if (!groups[g]) {
      groups[g] = [];
      groupOrder.push(g);
    }
    groups[g].push(spec);
  });

  // Render each group
  groupOrder.forEach(function(groupName) {
    var header = document.createElement('div');
    header.className = 'knob-group-header';
    header.textContent = groupName.charAt(0).toUpperCase() + groupName.slice(1);
    panel.appendChild(header);

    groups[groupName].forEach(function(spec) {
      panel.appendChild(buildKnobRow(spec));
    });
  });

  // Wire up input event listeners for staged-change highlighting
  _wireKnobInputListeners();
}

function buildKnobRow(spec) {
  // spec = {param_key, type, label, group, min, max, step, log_base, choices, current, state}
  var row = document.createElement('div');
  row.className = 'knob-row';
  row.dataset.param = spec.param_key;

  var label = document.createElement('span');
  label.className = 'knob-label';
  label.textContent = spec.label || spec.param_key;
  row.appendChild(label);

  if (spec.type === 'bool') {
    // Toggle switch
    var toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.className = 'knob-toggle';
    toggle.checked = !!spec.current;
    toggle.dataset.param = spec.param_key;
    row.appendChild(toggle);
  } else if (spec.type === 'choice') {
    // Dropdown
    var select = document.createElement('select');
    select.className = 'knob-select';
    select.dataset.param = spec.param_key;
    (spec.choices || []).forEach(function(c) {
      var opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      if (c === spec.current) opt.selected = true;
      select.appendChild(opt);
    });
    row.appendChild(select);
  } else if (spec.type === 'log_float') {
    // Log-scale slider
    var logMin = Math.log10(spec.min || 1e-7);
    var logMax = Math.log10(spec.max || 1.0);
    var logCurrent = (spec.current && spec.current > 0) ? Math.log10(spec.current) : logMin;

    var slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'knob-slider';
    slider.min = logMin;
    slider.max = logMax;
    slider.step = spec.step || 0.01;
    slider.value = logCurrent;
    slider.dataset.param = spec.param_key;
    slider.dataset.logScale = 'true';
    row.appendChild(slider);

    var valInput = document.createElement('input');
    valInput.type = 'text';
    valInput.className = 'knob-val';
    valInput.value = spec.current != null ? Number(spec.current).toExponential(2) : '';
    valInput.dataset.param = spec.param_key;
    row.appendChild(valInput);
  } else {
    // Linear slider (float, int)
    var slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'knob-slider';
    slider.min = spec.min != null ? spec.min : 0;
    slider.max = spec.max != null ? spec.max : 1;
    slider.step = spec.step || 0.01;
    slider.value = spec.current != null ? spec.current : 0;
    slider.dataset.param = spec.param_key;
    row.appendChild(slider);

    var valInput = document.createElement('input');
    valInput.type = 'text';
    valInput.className = 'knob-val';
    valInput.value = spec.current != null ? spec.current : '';
    valInput.dataset.param = spec.param_key;
    row.appendChild(valInput);
  }

  // State indicator
  if (spec.state && spec.state !== 'untouched' && spec.state !== 'verified') {
    var stateEl = document.createElement('span');
    stateEl.className = 'knob-state knob-state-' + spec.state;
    stateEl.textContent = spec.state;
    row.appendChild(stateEl);
  }

  return row;
}

function _wireKnobInputListeners() {
  var panel = document.getElementById('knobPanel');
  if (!panel) return;

  // Sliders: sync display value on input
  panel.querySelectorAll('.knob-slider').forEach(function(slider) {
    slider.addEventListener('input', function(e) {
      var param = e.target.dataset.param;
      var valEl = panel.querySelector('.knob-val[data-param="' + param + '"]');
      if (!valEl) return;
      if (e.target.dataset.logScale === 'true') {
        valEl.value = Math.pow(10, parseFloat(e.target.value)).toExponential(2);
      } else {
        valEl.value = parseFloat(e.target.value).toFixed(2);
      }
      _markStagedChanges();
    });
  });

  // Toggles and selects: mark staged changes on change
  panel.querySelectorAll('.knob-toggle, .knob-select').forEach(function(el) {
    el.addEventListener('change', function() { _markStagedChanges(); });
  });
}

/**
 * Read current value from a dynamic control row.
 */
function _readKnobValue(row) {
  var slider = row.querySelector('.knob-slider');
  var toggle = row.querySelector('.knob-toggle');
  var select = row.querySelector('.knob-select');

  if (toggle) return toggle.checked;
  if (select) return select.value;
  if (slider) {
    if (slider.dataset.logScale === 'true') {
      return Math.pow(10, parseFloat(slider.value));
    }
    return parseFloat(slider.value);
  }
  return undefined;
}

/**
 * Build the module -> param_key mapping for apply commands.
 * Groups determine which API endpoint to use:
 *   "optimizer" -> POST /api/opt/set
 *   "loss" -> POST /api/loss/set
 *   Other groups -> POST /api/opt/set (generic param set)
 */
function _getControlSpec(paramKey) {
  var specs = (S.config && S.config.controls) || [];
  for (var i = 0; i < specs.length; i++) {
    if (specs[i].param_key === paramKey) return specs[i];
  }
  return null;
}

function _markStagedChanges() {
  var panel = document.getElementById('knobPanel');
  if (!panel) return;
  var threshold = (S.config && S.config.ui && S.config.ui.staged_change_threshold) || 0.005;
  panel.querySelectorAll('.knob-row[data-param]').forEach(function(row) {
    var param = row.dataset.param;
    var current = _readKnobValue(row);
    var applied = _appliedKnobs[param];

    if (applied === undefined || current === undefined) {
      row.classList.remove('staged');
      return;
    }
    if (typeof current === 'boolean') {
      if (current !== applied) row.classList.add('staged');
      else row.classList.remove('staged');
    } else if (typeof current === 'string') {
      if (current !== applied) row.classList.add('staged');
      else row.classList.remove('staged');
    } else {
      var denom = Math.max(Math.abs(applied), 1e-12);
      if (Math.abs(current - applied) / denom > threshold) {
        row.classList.add('staged');
      } else {
        row.classList.remove('staged');
      }
    }
  });
}

function _snapshotAppliedKnobs() {
  var panel = document.getElementById('knobPanel');
  if (!panel) return;
  panel.querySelectorAll('.knob-row[data-param]').forEach(function(row) {
    var param = row.dataset.param;
    var val = _readKnobValue(row);
    if (val !== undefined) _appliedKnobs[param] = val;
  });
  // Clear all staged highlights
  document.querySelectorAll('.knob-row.staged').forEach(function(el) { el.classList.remove('staged'); });
}

function debounceApply(fn, delay) {
  if (_applyDebounceTimer) clearTimeout(_applyDebounceTimer);
  _applyDebounceTimer = setTimeout(fn, delay || 300);
}

function _clearTrainingState() {
  S.metricsData = {};
  S.appliedData = [];
  S.metricNames = new Set();
  S.metricColors = {};
  S.colorIdx = 0;
  S.latestMetrics = {};
  S.recipeEntries = [];
  S.chatHistory = [];
  S.alerts = [];
  _healthEma = null;
  _appliedKnobs = {};
  // Reset WS slider sync flag so next training session re-syncs
  if (typeof _slidersInitialized !== 'undefined') _slidersInitialized = false;
  // Clear focus/zoom state
  if (S.focusMetric) {
    S.focusMetric = null;
    document.body.classList.remove('metric-focus-mode');
  }
  // Clear pinned metric cards
  S.pinnedMetrics = new Set();
  if (S.metricCardCharts) {
    Object.keys(S.metricCardCharts).forEach(function(k) {
      if (S.metricCardCharts[k]) S.metricCardCharts[k].destroy();
    });
  }
  S.metricCardCharts = {};
  var mcContainer = $('#metricCards');
  if (mcContainer) mcContainer.innerHTML = '';
  // Clear forecast cache
  _forecastCache = {};
  _highlightedMutationStep = null;
  // Clear metric toggle state so old metric names don't persist
  if (typeof _metricToggleState !== 'undefined') _metricToggleState = {};
  if (typeof _metricDropdownShowAll !== 'undefined') _metricDropdownShowAll = false;
  if (typeof _lastMetricCount !== 'undefined') _lastMetricCount = 0;
  // Reset run-reset detection
  if (typeof _lastSeenMaxStep !== 'undefined') _lastSeenMaxStep = 0;
  // Reset step range to "All" for fresh run
  if (typeof _chartStepRange !== 'undefined') _chartStepRange = 'all';
  if (typeof _updateRangeButtons === 'function') _updateRangeButtons();
  clearTimelineDedup();
  updateMetricToggles();
  updateChart();
  setHealth(50, 'Reset \u2014 ready');
  var tl = $('#timelineList');
  if (tl) tl.innerHTML = '';
  var saveDetail = document.querySelector('.save-recipe-detail');
  if (saveDetail) saveDetail.remove();
  $('#mutationCount').textContent = '0';
  $('#stepValue').textContent = '--';
}

function _updateConfigControls(configId) {
  // No-op: controls are now generated dynamically from actuator metadata.
  // Kept as a stub because initControls and launcher still reference it.
}

function initControls() {
  // Dynamic controls — build from config if available
  if (S.config && S.config.controls && S.config.controls.length) {
    buildControls(S.config.controls);
  }

  // Apply — dynamic: collect all changed params from knobPanel
  $('#btnApply').addEventListener('click', function() {
    var btn = $('#btnApply');
    btn.textContent = 'Queued...';
    btn.disabled = true;
    debounceApply(async function() {
      var anyChanged = false;
      var threshold = (S.config && S.config.ui && S.config.ui.staged_change_threshold) || 0.005;

      function _isDiff(current, baseline) {
        if (baseline === undefined) return true;
        if (typeof current === 'boolean') return current !== baseline;
        if (typeof current === 'string') return current !== baseline;
        var denom = Math.max(Math.abs(baseline), 1e-12);
        return Math.abs(current - baseline) / denom > threshold;
      }

      // Collect changed params grouped by module
      var changedByGroup = {};  // group -> {param_key: value}
      var panel = document.getElementById('knobPanel');
      if (panel) {
        panel.querySelectorAll('.knob-row[data-param]').forEach(function(row) {
          var param = row.dataset.param;
          var current = _readKnobValue(row);
          if (current === undefined) return;
          if (!_isDiff(current, _appliedKnobs[param])) return;

          var spec = _getControlSpec(param);
          var group = (spec && spec.group) || 'optimizer';
          if (!changedByGroup[group]) changedByGroup[group] = {};
          changedByGroup[group][param] = current;
        });
      }

      // Send commands per group
      var groups = Object.keys(changedByGroup);
      for (var gi = 0; gi < groups.length; gi++) {
        var group = groups[gi];
        var params = changedByGroup[group];
        if (group === 'optimizer') {
          await api('POST', '/api/opt/set', {params: params});
        } else if (group === 'loss') {
          await api('POST', '/api/loss/set', {params: params});
        } else {
          // Generic: send each param individually as opt set
          var pkeys = Object.keys(params);
          for (var pi = 0; pi < pkeys.length; pi++) {
            var p = {};
            p[pkeys[pi]] = params[pkeys[pi]];
            await api('POST', '/api/opt/set', {params: p});
          }
        }
        anyChanged = true;
      }

      if (!anyChanged) {
        btn.textContent = 'No changes';
        btn.disabled = false;
        setTimeout(function() { btn.textContent = 'Apply'; }, 1500);
        return;
      }

      _snapshotAppliedKnobs();
      updateChart();
      // Snapshot forecast at mutation point
      var allSteps = [];
      Object.values(S.metricsData).forEach(function(a) {
        a.forEach(function(p) { allSteps.push(p.step); });
      });
      var currentStep = allSteps.length ? Math.max.apply(null, allSteps) : 0;
      recordMutationForecast(currentStep);
      setTimeout(fetchAllForecasts, 2000);
      btn.textContent = 'Apply';
      btn.disabled = false;
    }, 300);
  });

  // Schedule
  $('#btnSchedule').addEventListener('click', function() { openModal('modalSchedule'); });
  $('#btnScheduleSubmit').addEventListener('click', async function() {
    var step = parseInt($('#schedStep').value);
    if (!step || step <= 0) return;
    // Collect current values from all dynamic controls
    var panel = document.getElementById('knobPanel');
    var params = {};
    if (panel) {
      panel.querySelectorAll('.knob-row[data-param]').forEach(function(row) {
        var param = row.dataset.param;
        var val = _readKnobValue(row);
        if (val !== undefined) params[param] = val;
      });
    }
    if (Object.keys(params).length === 0) {
      alert('No control values to schedule. Adjust controls first.');
      return;
    }
    // Determine module from first param's group
    var firstKey = Object.keys(params)[0];
    var spec = _getControlSpec(firstKey);
    var module = (spec && spec.group === 'loss') ? 'loss' : 'opt';
    await api('POST', '/api/schedule', {at_step: step, module: module, op: 'set_params', params: params});
    closeModal('modalSchedule');
  });

  // Freeze
  $('#freezeSelect').addEventListener('change', async function(e) {
    var mode = e.target.value;
    if (mode !== 'off') {
      if (!confirm('Freeze mode "' + mode + '" will lock parameters. This cannot be easily undone. Continue?')) {
        e.target.value = 'off';
        return;
      }
    }
    await api('POST', '/api/freeze', {mode: mode});
    if (mode !== 'off') {
      e.target.disabled = true;
      e.target.style.opacity = '0.5';
      // Show unlock button
      var unlockBtn = document.getElementById('btnFreezeUnlock');
      if (!unlockBtn) {
        unlockBtn = document.createElement('button');
        unlockBtn.id = 'btnFreezeUnlock';
        unlockBtn.className = 'btn btn-sm';
        unlockBtn.textContent = 'Unlock';
        unlockBtn.style.cssText = 'font-size:8px;color:var(--red);margin-left:4px;';
        unlockBtn.addEventListener('click', async function() {
          if (!confirm('Are you sure you want to unlock freeze? This overrides safety controls.')) return;
          await api('POST', '/api/freeze', {mode: 'off'});
          e.target.disabled = false;
          e.target.style.opacity = '1';
          e.target.value = 'off';
          unlockBtn.remove();
        });
        e.target.parentNode.insertBefore(unlockBtn, e.target.nextSibling);
      }
    }
  });

  // Autopilot
  $('#autopilotMode').addEventListener('change', async function(e) {
    var mode = e.target.value;
    var res = await api('POST', '/api/autopilot/mode', {mode: mode});
    if (res && res.error) {
      alert('Failed to set mode: ' + (res.detail || res.error || 'unknown error'));
      e.target.value = 'off';
    }
    _updateAIConfigVisibility(mode);
    pollAutopilotStatus();
    if (mode.startsWith('ai_')) pollAIStatus();
  });
  setInterval(pollAutopilotStatus, 3000);

  // AI key metric dropdown + direction mode
  var aiKeyMetricEl = document.getElementById('aiKeyMetric');
  var aiKeyMetricModeEl = document.getElementById('aiKeyMetricMode');
  if (aiKeyMetricEl) {
    aiKeyMetricEl.addEventListener('change', async function(e) {
      var mode = aiKeyMetricModeEl ? aiKeyMetricModeEl.value : 'auto';
      await api('POST', '/api/autopilot/ai/key_metric', {metric: e.target.value, mode: mode});
      pollAIStatus();
    });
  }
  if (aiKeyMetricModeEl) {
    aiKeyMetricModeEl.addEventListener('change', async function(e) {
      var metric = aiKeyMetricEl ? aiKeyMetricEl.value : 'val_loss';
      await api('POST', '/api/autopilot/ai/key_metric', {metric: metric, mode: e.target.value});
      pollAIStatus();
    });
  }

  // Training launcher — config selection
  var _trainConfigDescs = {
    simple: 'Single-task quadratic loss. Good for testing basic controls.',
    multitask: 'Two-task training with recipe-driven loss weight shifts.',
    finetune: 'Pretrained backbone finetune with freeze/unfreeze toggle.',
  };
  var _trainConfigDefaults = {
    simple: {max_steps: 500, step_delay: 0.15},
    multitask: {max_steps: 800, step_delay: 0.12},
    finetune: {max_steps: 600, step_delay: 0.12},
  };
  $('#trainConfig').addEventListener('change', function(e) {
    var id = e.target.value;
    $('#trainConfigDesc').textContent = _trainConfigDescs[id] || '';
    var defs = _trainConfigDefaults[id] || {};
    if (defs.max_steps) $('#trainMaxSteps').value = defs.max_steps;
    if (defs.step_delay) $('#trainStepDelay').value = defs.step_delay;
    _updateConfigControls(id);
    // Check for recipe and update toggle status
    var recipeStatus = document.getElementById('trainRecipeStatus');
    if (recipeStatus) {
      api('GET', '/api/recipe/').then(function(data) {
        if (data && data.entries && data.entries.length > 0) {
          recipeStatus.textContent = data.entries.length + ' entries available';
        } else {
          recipeStatus.textContent = 'no recipe loaded';
        }
      });
    }
  });

  // Initial config controls state
  _updateConfigControls($('#trainConfig').value);

  $('#btnTrainStart').addEventListener('click', async function() {
    var configId = $('#trainConfig').value;
    var maxSteps = parseInt($('#trainMaxSteps').value) || 800;
    var stepDelay = parseFloat($('#trainStepDelay').value) || 0.12;

    // Check if already running — prompt to kill
    var status = await api('GET', '/api/train/status');
    if (status && status.running) {
      var prev = (status.config && status.config.config_name) || 'training';
      if (!confirm('"' + prev + '" is still running. Stop it and start "' + configId + '"?')) return;
    }

    // Always reset before starting — gives a clean slate
    await api('POST', '/api/train/reset', {});
    _clearTrainingState();

    // If "load recipe" is checked, load the recipe before starting
    var loadRecipe = document.getElementById('trainLoadRecipe');
    if (loadRecipe && loadRecipe.checked) {
      var recipeStatus = document.getElementById('trainRecipeStatus');
      if (recipeStatus) recipeStatus.textContent = 'loading recipe...';
      var recipeData = await api('GET', '/api/recipe/');
      if (recipeData && recipeData.entries && recipeData.entries.length > 0) {
        if (recipeStatus) recipeStatus.textContent = recipeData.entries.length + ' entries loaded';
      } else {
        if (recipeStatus) recipeStatus.textContent = 'no recipe found';
      }
    }

    var seedInput = document.getElementById('trainSeed');
    var seedVal = seedInput && seedInput.value ? parseInt(seedInput.value) : null;
    var startBody = {
      config_id: configId,
      max_steps: maxSteps,
      step_delay: stepDelay,
    };
    if (seedVal !== null && !isNaN(seedVal)) startBody.seed = seedVal;
    var res = await api('POST', '/api/train/start', startBody);
    if (res && !res.error) {
      pollTrainStatus();
    } else if (res && res.error) {
      alert('Failed to start training: ' + res.error);
      pollTrainStatus();
    }
  });
  $('#btnTrainStop').addEventListener('click', async function() {
    var btn = $('#btnTrainStop');
    btn.disabled = true;
    btn.textContent = 'Stopping...';
    var res = await api('POST', '/api/train/stop');
    // Poll status after a brief delay to let the thread wind down
    setTimeout(function() {
      pollTrainStatus();
      btn.textContent = 'Stop';
    }, 500);
    // Second poll to catch slower shutdowns
    setTimeout(pollTrainStatus, 2000);
  });
  $('#btnTrainReset').addEventListener('click', async function() {
    if (!confirm('Reset will stop training and clear all data. Continue?')) return;
    var btn = $('#btnTrainReset');
    btn.disabled = true;
    btn.textContent = 'Resetting...';
    await api('POST', '/api/train/reset', {});
    _clearTrainingState();
    pollTrainStatus();
    btn.disabled = false;
    btn.textContent = 'Reset';
  });
  setInterval(pollTrainStatus, 5000);
  pollTrainStatus();

  // Fetch available configs from server
  api('GET', '/api/train/configs').then(function(res) {
    if (!res || !res.configs) return;
    var sel = $('#trainConfig');
    sel.innerHTML = '';
    res.configs.forEach(function(cfg) {
      var opt = document.createElement('option');
      opt.value = cfg.config_id;
      opt.textContent = cfg.name;
      sel.appendChild(opt);
      _trainConfigDescs[cfg.config_id] = cfg.description;
      _trainConfigDefaults[cfg.config_id] = cfg.defaults;
    });
    sel.value = 'multitask';
    $('#trainConfigDesc').textContent = _trainConfigDescs['multitask'] || '';
  });

  // Manifold method change
  $('#manifoldMethod').addEventListener('change', fetchManifold);

  // Theme
  $('#themeSelect').addEventListener('change', function(e) {
    setTheme(e.target.value);
  });

  // Hydrate controls from server state (works for both launcher and external training)
  hydrateControlsFromServer().then(function() { _snapshotAppliedKnobs(); });

  // Poll for controls periodically — actuator file may appear after training starts.
  // Keep polling until we get MORE controls than the defaults (lr + weight_decay = 2).
  var _controlsPollCount = 0;
  var _controlsPollMax = 30;  // stop after ~90 seconds
  setInterval(function() {
    _controlsPollCount++;
    if (_controlsPollCount > _controlsPollMax) return;
    var currentCount = (S.config && S.config.controls) ? S.config.controls.length : 0;
    api('GET', '/api/state/controls').then(function(state) {
      if (!state || !state.controls || !state.controls.length) return;
      // Only rebuild if we got MORE controls or controls changed
      if (state.controls.length <= currentCount && currentCount > 2) return;
      if (state.controls.length > currentCount || currentCount <= 2) {
        buildControls(state.controls);
        if (S.config) S.config.controls = state.controls;
        if (state.last_opt_params) syncSlidersFromApplied(state.last_opt_params);
        if (state.last_loss_params) syncSlidersFromApplied(state.last_loss_params);
        _snapshotAppliedKnobs();
        // Stop polling once we have real controls (more than defaults)
        if (state.controls.length > 2) _controlsPollCount = _controlsPollMax + 1;
      }
    });
  }, 3000);
}

/* ================================================================ */
/* Capabilities-aware UI adaptation                                  */
/* ================================================================ */

async function loadCapabilities() {
  try {
    var caps = await api('GET', '/api/capabilities');
    if (!caps || !caps.detected) return;
    S.capabilities = caps;
    // Capabilities are now informational only — controls are generated from config.
  } catch(e) { /* ignore */ }
}

function syncSlidersFromApplied(params) {
  if (!params || typeof params !== 'object') return;
  var panel = document.getElementById('knobPanel');
  if (!panel) return;

  Object.keys(params).forEach(function(k) {
    var value = params[k];
    if (typeof value !== 'number' && typeof value !== 'boolean' && typeof value !== 'string') return;

    // Update baseline
    _appliedKnobs[k] = value;

    // Also check aliases: weight_decay -> weight_decay param key
    var paramKey = k;

    // Find the matching dynamic control row
    var row = panel.querySelector('.knob-row[data-param="' + paramKey + '"]');
    if (!row) return;

    var slider = row.querySelector('.knob-slider');
    var valEl = row.querySelector('.knob-val');
    var toggle = row.querySelector('.knob-toggle');
    var select = row.querySelector('.knob-select');

    if (toggle && typeof value === 'boolean') {
      toggle.checked = value;
    } else if (select && typeof value === 'string') {
      select.value = value;
    } else if (slider && typeof value === 'number') {
      if (slider.dataset.logScale === 'true' && value > 0) {
        slider.value = Math.log10(value);
        if (valEl) valEl.value = value.toExponential(2);
      } else {
        slider.value = value;
        if (valEl) valEl.value = parseFloat(value).toFixed(2);
      }
    }
  });

  _markStagedChanges();
}

async function hydrateControlsFromServer() {
  try {
    var state = await api('GET', '/api/state/controls');
    if (!state) return;

    // Build dynamic controls from server-provided controls list
    if (state.controls && state.controls.length) {
      buildControls(state.controls);
      // Update S.config.controls so Apply handler can look up specs
      if (S.config) S.config.controls = state.controls;
    }

    // Demo mode gate: hide entire Training card when not in demo mode
    if (state.demo_mode === false) {
      var trainPanel = document.getElementById('trainPanel');
      if (trainPanel) {
        var trainCard = trainPanel.closest('.card');
        if (trainCard) trainCard.style.display = 'none';
      }
    }

    // Sync sliders from last applied params
    if (state.last_opt_params) syncSlidersFromApplied(state.last_opt_params);
    if (state.last_loss_params) syncSlidersFromApplied(state.last_loss_params);

    // Module activity detection — controls are now dynamic, no CSS class hiding needed

    // External training: hide demo config dropdown, show attached label
    if (state.is_external) {
      var trainConfig = document.getElementById('trainConfig');
      var trainConfigDesc = document.getElementById('trainConfigDesc');
      if (trainConfig) trainConfig.style.display = 'none';
      if (trainConfigDesc) trainConfigDesc.textContent = 'External Training (attached)';
    }

    // Sync config from run.json (for non-launcher runs)
    var runCfg = state.run_config || {};
    if (runCfg.config_id) {
      var sel = document.getElementById('trainConfig');
      // Only sync if it's a known config
      if (sel) {
        var found = false;
        for (var i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === runCfg.config_id) { found = true; break; }
        }
        if (found) {
          sel.value = runCfg.config_id;
          _updateConfigControls(runCfg.config_id);
        }
      }
      if (runCfg.max_steps) {
        var msEl = document.getElementById('trainMaxSteps');
        if (msEl) msEl.value = runCfg.max_steps;
      }
    }

    // Sync step counter
    if (state.latest_step) {
      var stepEl = document.getElementById('stepValue');
      if (stepEl) stepEl.textContent = state.latest_step;
    }

    // Sync autopilot mode
    if (state.autopilot_mode && state.autopilot_mode !== 'off') {
      var modeSelect = document.getElementById('autopilotMode');
      if (modeSelect) {
        modeSelect.value = state.autopilot_mode;
        _updateAIConfigVisibility(state.autopilot_mode);
      }
    }
  } catch (e) { /* ignore hydration errors */ }
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('hotcb-theme', theme);
  // Update Three.js scene backgrounds if they exist
  var sceneBg = getComputedStyle(document.documentElement).getPropertyValue('--scene-bg').trim() || '#0a1018';
  var bgColor = parseInt(sceneBg.replace('#', ''), 16);
  if (S.manifoldCtx && S.manifoldCtx.scene) {
    S.manifoldCtx.scene.background = new THREE.Color(bgColor);
  }
  if (S.featureCtx && S.featureCtx.scene) {
    S.featureCtx.scene.background = new THREE.Color(bgColor);
  }
  // Update chart tooltip colors for new theme
  if (S.chartInstance) {
    var cs = getComputedStyle(document.documentElement);
    S.chartInstance.options.plugins.tooltip.backgroundColor = cs.getPropertyValue('--bg-card').trim();
    S.chartInstance.options.plugins.tooltip.borderColor = cs.getPropertyValue('--border-bright').trim();
    S.chartInstance.options.scales.x.grid.color = cs.getPropertyValue('--border').trim();
    S.chartInstance.options.scales.y.grid.color = cs.getPropertyValue('--border').trim();
    S.chartInstance.options.scales.x.ticks.color = cs.getPropertyValue('--text-muted').trim();
    S.chartInstance.options.scales.y.ticks.color = cs.getPropertyValue('--text-muted').trim();
    S.chartInstance.update('none');
  }
}

/**
 * Infer whether a metric should be minimized or maximized from its name.
 * Mirrors the server-side infer_metric_direction().
 */
function inferMetricDirection(name) {
  var low = name.toLowerCase();
  var minPats = ['loss', 'error', 'err', 'perplexity', 'ppl', 'mse', 'mae', 'rmse',
                 'cer', 'wer', 'fid', 'divergence', 'regret', 'cost'];
  var maxPats = ['accuracy', 'acc', 'f1', 'auc', 'auroc', 'recall', 'precision',
                 'score', 'bleu', 'rouge', 'meteor', 'iou', 'dice', 'map',
                 'reward', 'return', 'r2', 'correlation', 'similarity',
                 'alignment', 'coherence', 'fluency'];
  for (var i = 0; i < minPats.length; i++) { if (low.indexOf(minPats[i]) >= 0) return 'min'; }
  for (var i = 0; i < maxPats.length; i++) { if (low.indexOf(maxPats[i]) >= 0) return 'max'; }
  return 'min';  // default
}

function computeHealth() {
  // Try to find a primary metric: prefer train_loss, then any loss, then first metric
  var lossData = S.metricsData['train_loss'] || S.metricsData['loss'] || [];
  var primaryMetric = 'train_loss';
  var primaryDir = 'min';
  if (!lossData.length) {
    // Fall back to first available metric
    var names = Object.keys(S.metricsData);
    for (var i = 0; i < names.length; i++) {
      if (S.metricsData[names[i]].length >= 5) {
        lossData = S.metricsData[names[i]];
        primaryMetric = names[i];
        primaryDir = inferMetricDirection(names[i]);
        break;
      }
    }
  }
  if (lossData.length < 5) { setHealth(50, 'Insufficient data'); return; }
  var recent = lossData.slice(-30);
  var first = recent[0].value;
  var last = recent[recent.length - 1].value;
  var trend = (last - first) / Math.max(Math.abs(first), 1e-8);

  // Flip trend for maximize metrics (positive trend = improvement)
  var effectiveTrend = (primaryDir === 'max') ? -trend : trend;

  var rawScore = 70;
  if (effectiveTrend < -0.05) rawScore = 90;
  else if (effectiveTrend < 0) rawScore = 80;
  else if (effectiveTrend < 0.05) rawScore = 60;
  else if (effectiveTrend < 0.2) rawScore = 40;
  else rawScore = 20;

  // Check for overfitting via val metric
  var valMetric = S.metricsData['val_loss'] || S.metricsData['val_error'];
  if (valMetric && valMetric.length >= 3) {
    var vRecent = valMetric.slice(-5);
    // For minimize metrics, val going up is bad
    if (vRecent[vRecent.length-1].value > vRecent[0].value * 1.2) rawScore = Math.min(rawScore, 35);
  }

  // EMA smoothing
  if (_healthEma === null) {
    _healthEma = rawScore;
  } else {
    _healthEma = 0.85 * _healthEma + 0.15 * rawScore;
  }

  // Clamp: don't jump more than 5 points per update
  var prev = S.healthScore || _healthEma;
  var score = Math.round(_healthEma);
  if (Math.abs(score - prev) > 5) {
    score = prev + (score > prev ? 5 : -5);
  }
  score = Math.max(0, Math.min(100, score));

  var desc = score >= 80 ? 'Healthy \u2014 loss improving' :
             score >= 60 ? 'Stable \u2014 minimal change' :
             score >= 40 ? 'Plateau \u2014 consider intervention' :
             'Warning \u2014 loss increasing';
  setHealth(score, desc);
  updateHealthMetrics();
}

function updateHealthMetrics() {
  var el = document.getElementById('healthMetrics');
  if (!el) return;
  var keys = ['train_loss', 'val_loss', 'lr', 'accuracy', 'val_accuracy', 'grad_norm'];
  el.innerHTML = '';
  keys.forEach(function(k) {
    var v = S.latestMetrics[k];
    if (v === undefined) return;
    var div = document.createElement('div');
    div.style.cssText = 'display:flex;justify-content:space-between;';
    div.innerHTML = '<span style="color:var(--text-muted)">' + k + '</span><span style="color:var(--text-primary)">' + fmtNum(v) + '</span>';
    el.appendChild(div);
  });
}

function setHealth(score, desc) {
  S.healthScore = score;

  // Determine status tier
  var icon, statusColor, bgTint;
  if (score > 70) {
    icon = '\u2713'; statusColor = 'var(--green, #4ade80)';
    bgTint = 'rgba(74, 222, 128, 0.07)';
  } else if (score >= 40) {
    icon = '\u26A0'; statusColor = 'var(--yellow, #facc15)';
    bgTint = 'rgba(250, 204, 21, 0.07)';
  } else if (score >= 20) {
    icon = '\u26A1'; statusColor = 'var(--orange, #fb923c)';
    bgTint = 'rgba(251, 146, 60, 0.07)';
  } else {
    icon = '\u2715'; statusColor = 'var(--red, #f87171)';
    bgTint = 'rgba(248, 113, 113, 0.08)';
  }

  // Score display with icon
  $('#healthScore').innerHTML = '<span style="margin-right:4px">' + icon + '</span>' + score;
  $('#healthScore').style.color = statusColor;

  // Health bar fill
  var fill = $('#healthFill');
  fill.style.width = score + '%';
  fill.style.background = statusColor;

  // Color-coded card background and border
  var card = $('#healthCard');
  if (card) {
    card.style.borderLeft = '3px solid ' + statusColor;
    card.style.background = 'linear-gradient(135deg, ' + bgTint + ', transparent 60%)';
  }

  $('#healthDesc').textContent = desc;
}

var _lastSyncedConfigId = null;
var _wasTrainingRunning = false;
var _runConfigName = '';

async function pollTrainStatus() {
  try {
    var res = await api('GET', '/api/train/status');
    if (!res) return;
    var el = $('#trainStatus');
    var btnStart = $('#btnTrainStart');
    var btnStop = $('#btnTrainStop');
    if (res.running) {
      _wasTrainingRunning = true;
      var cfg = res.config || {};
      _runConfigName = cfg.config_name || cfg.config_id || '';
      var info = 'Running: ' + (cfg.config_name || cfg.config_id || '?');
      if (cfg.max_steps) info += ' (' + cfg.max_steps + ' steps)';
      if (cfg.seed !== undefined) info += ' seed=' + cfg.seed;
      el.textContent = info;
      el.style.color = 'var(--green, #4ade80)';
      btnStart.disabled = true;
      btnStop.disabled = false;

      // Sync controls to match the running config (once per config change)
      if (cfg.config_id && cfg.config_id !== _lastSyncedConfigId) {
        _lastSyncedConfigId = cfg.config_id;
        var sel = document.getElementById('trainConfig');
        if (sel && sel.value !== cfg.config_id) {
          sel.value = cfg.config_id;
          _updateConfigControls(cfg.config_id);
          var desc = document.getElementById('trainConfigDesc');
          if (desc && cfg.config_name) desc.textContent = cfg.config_name;
        }
        if (cfg.max_steps) {
          var msEl = document.getElementById('trainMaxSteps');
          if (msEl) msEl.value = cfg.max_steps;
        }
        if (cfg.step_delay !== undefined) {
          var sdEl = document.getElementById('trainStepDelay');
          if (sdEl) sdEl.value = cfg.step_delay;
        }
        if (cfg.seed !== undefined) {
          var seedInput = document.getElementById('trainSeed');
          if (seedInput) seedInput.value = cfg.seed;
        }
      }
    } else {
      // Detect running → stopped transition
      if (_wasTrainingRunning) {
        _wasTrainingRunning = false;
        showRunSummary(_runConfigName);
      }
      el.textContent = 'Stopped';
      el.style.color = 'var(--text-muted)';
      btnStart.disabled = false;
      btnStop.disabled = true;
      _lastSyncedConfigId = null;
    }
  } catch (e) { /* ignore poll errors */ }
}

/* ================================================================ */
/* Save applied mutations as recipe                                  */
/* ================================================================ */

function initSaveAsRecipe() {
  var btn = $('#btnSaveAsRecipe');
  if (!btn) return;
  btn.addEventListener('click', async function() {
    btn.disabled = true;
    btn.textContent = 'Saving...';
    var res = await api('POST', '/api/applied/save-as-recipe');
    if (res && res.status === 'saved') {
      // Show success details in a panel below the button
      var container = btn.parentElement;
      var detail = container.querySelector('.save-recipe-detail');
      if (!detail) {
        detail = document.createElement('div');
        detail.className = 'save-recipe-detail';
        detail.style.cssText = 'font-size:10px;font-family:var(--font-mono);margin-top:6px;padding:6px 8px;' +
          'background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);border-radius:4px;';
        container.appendChild(detail);
      }
      var savedPath = res.path || 'hotcb.recipe.jsonl';
      var savedCount = res.count || 0;
      detail.innerHTML =
        '<div style="color:var(--green);font-weight:700;margin-bottom:3px">Saved ' + savedCount + ' entries</div>' +
        '<div style="color:var(--text-secondary);margin-bottom:5px;word-break:break-all">' + savedPath + '</div>' +
        '<button class="btn btn-sm btn-accent" id="btnViewInRecipeEditor">View in Recipe Editor</button>';

      detail.querySelector('#btnViewInRecipeEditor').addEventListener('click', function() {
        fetchRecipe();  // reload recipe data
        var tab = document.querySelector('.tab[data-tab="recipe-editor"]');
        if (tab) tab.click();
        detail.remove();
      });

      btn.textContent = 'Saved (' + savedCount + ' entries)';
      setTimeout(function() { btn.textContent = 'Save as Recipe'; btn.disabled = false; }, 3000);
    } else {
      btn.textContent = 'Failed';
      setTimeout(function() { btn.textContent = 'Save as Recipe'; btn.disabled = false; }, 2000);
    }
  });
}

/* ================================================================ */
/* Autopilot Status Polling                                          */
/* ================================================================ */

var _lastAutopilotCount = 0;

function _updateAIConfigVisibility(mode) {
  var section = document.getElementById('aiConfigSection');
  if (section) {
    section.style.display = (mode === 'ai_suggest' || mode === 'ai_auto') ? 'block' : 'none';
  }
}

async function pollAIStatus() {
  try {
    var res = await api('GET', '/api/autopilot/ai/status');
    if (!res || !res.enabled) return;

    // Update cost info
    var costEl = document.getElementById('aiCostInfo');
    if (costEl) {
      costEl.textContent = '$' + (res.total_cost_usd || 0).toFixed(4) + ' / $' + ((res.config && res.config.budget_cap) || 5).toFixed(2);
    }
    var callEl = document.getElementById('aiCallCount');
    if (callEl) {
      callEl.textContent = (res.call_count || 0) + ' calls';
    }

    // Run info
    var runEl = document.getElementById('aiRunInfo');
    var runNumEl = document.getElementById('aiRunNumber');
    if (runEl && runNumEl) {
      runEl.style.display = 'block';
      runNumEl.textContent = 'Run ' + (res.run_number || 1) + '/' + (res.max_runs || 3);
    }

    // Populate key metric dropdown with available metrics
    var keyMetricEl = document.getElementById('aiKeyMetric');
    if (keyMetricEl) {
      // Fetch metric names and populate
      var metricNames = await api('GET', '/api/metrics/names');
      if (metricNames && metricNames.names) {
        var currentVal = keyMetricEl.value;
        keyMetricEl.innerHTML = '';
        metricNames.names.forEach(function(n) {
          var opt = document.createElement('option');
          opt.value = n;
          opt.textContent = n;
          keyMetricEl.appendChild(opt);
        });
        keyMetricEl.value = res.key_metric || currentVal || 'val_loss';
      }
      // Sync direction mode
      var modeEl = document.getElementById('aiKeyMetricMode');
      if (modeEl && res.key_metric_mode) {
        modeEl.value = res.key_metric_mode;
      }
    }

    // Fetch latest AI reasoning
    var histRes = await api('GET', '/api/autopilot/ai/history?last_n=1');
    if (histRes && histRes.decisions && histRes.decisions.length > 0) {
      var latest = histRes.decisions[histRes.decisions.length - 1];
      var reasoningPanel = document.getElementById('aiReasoningPanel');
      var reasoningEl = document.getElementById('aiReasoning');
      if (reasoningPanel && reasoningEl && latest.reasoning) {
        reasoningPanel.style.display = 'block';
        reasoningEl.textContent = latest.reasoning;
      }
    }
  } catch (e) { /* ignore */ }
}

async function pollAutopilotStatus() {
  try {
    var res = await api('GET', '/api/autopilot/status');
    if (!res) return;
    var panel = $('#autopilotPanel');
    var statusEl = $('#autopilotStatus');
    if (!panel || !statusEl) return;

    var mode = res.mode || 'off';
    var rulesCount = res.rules_count || 0;
    var historyCount = res.history_count || 0;
    var recent = res.recent_actions || [];

    // Sync mode dropdown
    var modeSelect = document.getElementById('autopilotMode');
    if (modeSelect && modeSelect.value !== mode) {
      modeSelect.value = mode;
      _updateAIConfigVisibility(mode);
    }

    if (mode === 'off') {
      statusEl.innerHTML = '<span style="color:var(--text-muted)">Autopilot is off. Select a mode to enable.</span>';
      var existing = panel.querySelectorAll('.autopilot-action');
      existing.forEach(function(el) { el.remove(); });
      _lastAutopilotCount = 0;
      return;
    }

    var modeLabels = {
      'suggest': 'Suggesting',
      'auto': 'Auto-applying',
      'ai_suggest': 'AI Suggesting',
      'ai_auto': 'AI Auto-applying',
    };
    var modeLabel = modeLabels[mode] || mode;
    statusEl.innerHTML = '<span style="color:var(--cyan)">' + modeLabel + '</span> &middot; ' +
      rulesCount + ' rules &middot; ' + historyCount + ' action(s)';

    // Poll AI status if in AI mode
    if (mode === 'ai_suggest' || mode === 'ai_auto') {
      pollAIStatus();
    }

    if (historyCount !== _lastAutopilotCount) {
      _lastAutopilotCount = historyCount;
      var existing = panel.querySelectorAll('.autopilot-action');
      existing.forEach(function(el) { el.remove(); });
      var toShow = recent.slice().reverse().slice(0, 5);
      toShow.forEach(function(action) {
        var div = document.createElement('div');
        div.className = 'autopilot-action';
        div.style.cssText = 'font-size:10px;padding:4px 0;border-top:1px solid var(--border);margin-top:4px;';
        var isAI = (action.rule_id || '').startsWith('ai:');
        var statusColor = action.status === 'applied' ? 'var(--green, #4ade80)' :
                          action.status === 'proposed' ? 'var(--yellow, #facc15)' :
                          action.status === 'alert' ? 'var(--orange, #fb923c)' : 'var(--text-muted)';
        var statusLabel = isAI ? 'AI ' + action.status : action.status;
        var badge = '<span class="ap-badge" style="color:' + statusColor + ';font-weight:600;text-transform:uppercase;font-size:9px">' +
                    statusLabel + '</span>';
        var step = action.step || '?';
        var ruleId = action.rule_id || '?';
        var condition = action.condition_met || '';
        if (condition.length > 80) condition = condition.substring(0, 77) + '...';
        var fullCondition = action.condition_met || '';
        div.innerHTML = badge + ' <span style="color:var(--text-muted)">step ' + step + '</span> ' +
                        '<span style="color:var(--cyan)">' + ruleId + '</span><br>' +
                        '<span class="ap-condition" style="color:var(--text-muted);cursor:help" title="' +
                        fullCondition.replace(/"/g, '&quot;') + '">' + condition + '</span>';

        // Add "Apply" button for proposed (suggest-mode) actions
        if (action.status === 'proposed' && action.action_id) {
          var applyBtn = document.createElement('button');
          applyBtn.className = 'btn btn-sm';
          applyBtn.textContent = 'Apply';
          applyBtn.style.cssText = 'margin-top:3px;font-size:9px;padding:1px 8px;background:var(--cyan,#22d3ee);color:#000;border:none;border-radius:3px;cursor:pointer;';
          applyBtn.addEventListener('click', function() {
            applyBtn.disabled = true;
            applyBtn.textContent = 'Applying...';
            api('POST', '/api/autopilot/accept/' + action.action_id).then(function(res) {
              if (res && res.status === 'applied') {
                applyBtn.textContent = 'Applied';
                applyBtn.style.background = 'var(--green, #4ade80)';
                // Update badge color
                var badgeEl = div.querySelector('.ap-badge');
                if (badgeEl) {
                  badgeEl.textContent = 'applied';
                  badgeEl.style.color = 'var(--green, #4ade80)';
                }
                _lastAutopilotCount = 0; // force refresh on next poll
              } else {
                applyBtn.textContent = 'Failed';
                applyBtn.disabled = false;
              }
            });
          });
          div.appendChild(applyBtn);
        }

        panel.appendChild(div);
      });
    }
  } catch (e) { /* ignore poll errors */ }
}

/* ================================================================ */
/* Callback Diagnostics                                              */
/* ================================================================ */

async function fetchCallbacks() {
  var res = await api('GET', '/api/cb/list');
  if (!res || !res.callbacks) return;
  var list = $('#cbList');
  var status = $('#cbStatus');
  if (!list) return;

  if (res.callbacks.length === 0) {
    status.textContent = 'No callbacks registered';
    list.innerHTML = '';
    return;
  }
  status.textContent = res.callbacks.length + ' callback(s)';
  list.innerHTML = '';
  res.callbacks.forEach(function(cb) {
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid var(--border);font-size:11px;';
    var dot = document.createElement('span');
    dot.style.cssText = 'width:8px;height:8px;border-radius:50%;flex-shrink:0;background:' + (cb.enabled ? 'var(--green, #4ade80)' : 'var(--text-muted)');
    var label = document.createElement('span');
    label.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    label.textContent = cb.id;
    if (cb.params && cb.params.path) {
      label.title = cb.params.path;
    }
    var toggleBtn = document.createElement('button');
    toggleBtn.className = 'btn btn-sm';
    toggleBtn.textContent = cb.enabled ? 'Disable' : 'Enable';
    toggleBtn.style.fontSize = '9px';
    toggleBtn.addEventListener('click', function() {
      var op = cb.enabled ? 'disable' : 'enable';
      api('POST', '/api/cb/' + encodeURIComponent(cb.id) + '/' + op).then(fetchCallbacks);
    });
    var unloadBtn = document.createElement('button');
    unloadBtn.className = 'btn btn-sm';
    unloadBtn.textContent = 'Unload';
    unloadBtn.style.cssText = 'font-size:9px;color:var(--red, #f87171);';
    unloadBtn.addEventListener('click', function() {
      if (confirm('Unload callback "' + cb.id + '"?')) {
        api('POST', '/api/cb/' + encodeURIComponent(cb.id) + '/unload').then(fetchCallbacks);
      }
    });
    row.appendChild(dot);
    row.appendChild(label);
    row.appendChild(toggleBtn);
    row.appendChild(unloadBtn);
    list.appendChild(row);
  });
}

function initCallbackDiagnostics() {
  var btn = $('#btnCbRefresh');
  if (btn) btn.addEventListener('click', fetchCallbacks);

  // Load form toggle
  var btnToggle = $('#btnCbToggleLoad');
  var form = $('#cbLoadForm');
  if (btnToggle && form) {
    btnToggle.addEventListener('click', function() {
      form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });
  }

  // Cancel load form
  var btnCancel = $('#btnCbLoadCancel');
  if (btnCancel) {
    btnCancel.addEventListener('click', function() {
      form.style.display = 'none';
    });
  }

  // Submit load
  var btnSubmit = $('#btnCbLoadSubmit');
  if (btnSubmit) {
    btnSubmit.addEventListener('click', async function() {
      var cbId = $('#cbLoadId').value.trim();
      var cbPath = $('#cbLoadPath').value.trim();
      if (!cbId || !cbPath) {
        alert('Both Callback ID and Host Path are required.');
        return;
      }
      btnSubmit.disabled = true;
      btnSubmit.textContent = 'Loading...';
      var res = await api('POST', '/api/cb/load', {id: cbId, path: cbPath});
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Load';
      if (res && res.status === 'queued') {
        $('#cbLoadId').value = '';
        $('#cbLoadPath').value = '';
        form.style.display = 'none';
        fetchCallbacks();
      } else {
        alert('Failed to load callback.');
      }
    });
  }

  fetchCallbacks();
}
