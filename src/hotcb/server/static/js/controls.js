/**
 * hotcb dashboard — knobs, freeze, autopilot
 */

function lrFromSlider(v) { return Math.pow(10, parseFloat(v)); }
function sliderFromLr(lr) { return Math.log10(lr); }

var _applyDebounceTimer = null;
var _applyQueue = [];
var _lastApplyStep = 0;
var _healthEma = null;  // EMA-smoothed health score

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
  var multitaskEls = document.querySelectorAll('.multitask-only');
  var finetuneEls = document.querySelectorAll('.finetune-only');
  var singleLossEls = document.querySelectorAll('.single-loss-only');

  var isMultitask = configId === 'multitask';
  var isFinetune = configId === 'finetune';

  multitaskEls.forEach(function(el) { el.style.display = isMultitask ? '' : 'none'; });
  finetuneEls.forEach(function(el) { el.style.display = isFinetune ? '' : 'none'; });
  singleLossEls.forEach(function(el) { el.style.display = isMultitask ? 'none' : ''; });
}

function initControls() {
  // Knob sliders
  $('#knobLr').addEventListener('input', function(e) {
    var v = lrFromSlider(e.target.value);
    $('#knobLrVal').value = v.toExponential(2);
  });
  $('#knobWd').addEventListener('input', function(e) {
    var v = Math.pow(10, parseFloat(e.target.value));
    $('#knobWdVal').value = v.toExponential(2);
  });
  $('#knobLossW').addEventListener('input', function(e) {
    $('#knobLossWVal').value = parseFloat(e.target.value).toFixed(2);
  });
  $('#knobWeightA').addEventListener('input', function(e) {
    $('#knobWeightAVal').value = parseFloat(e.target.value).toFixed(2);
  });
  $('#knobWeightB').addEventListener('input', function(e) {
    $('#knobWeightBVal').value = parseFloat(e.target.value).toFixed(2);
  });

  // Apply
  $('#btnApply').addEventListener('click', function() {
    var btn = $('#btnApply');
    btn.textContent = 'Queued...';
    btn.disabled = true;
    debounceApply(async function() {
      var lr = lrFromSlider($('#knobLr').value);
      var wd = Math.pow(10, parseFloat($('#knobWd').value));
      await api('POST', '/api/opt/set', {params: {lr: lr, weight_decay: wd}});

      var configId = $('#trainConfig').value;
      if (configId === 'multitask') {
        var wa = parseFloat($('#knobWeightA').value);
        var wb = parseFloat($('#knobWeightB').value);
        await api('POST', '/api/loss/set', {params: {weight_a: wa, weight_b: wb}});
      } else {
        var lw = parseFloat($('#knobLossW').value);
        if (lw !== 1.0) await api('POST', '/api/loss/set', {params: {weight: lw}});
      }

      if (configId === 'finetune') {
        var frozen = $('#knobBackbone').value === '1';
        await api('POST', '/api/cb/set_params', {params: {backbone_frozen: frozen}});
      }
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
    var lr = lrFromSlider($('#knobLr').value);
    var wd = Math.pow(10, parseFloat($('#knobWd').value));
    await api('POST', '/api/schedule', {at_step: step, module: 'opt', op: 'set_params', params: {lr: lr, weight_decay: wd}});
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
    await api('POST', '/api/autopilot/mode', {mode: e.target.value});
    pollAutopilotStatus();
  });
  setInterval(pollAutopilotStatus, 3000);

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
    }
  });
  $('#btnTrainStop').addEventListener('click', async function() {
    var btn = $('#btnTrainStop');
    btn.disabled = true;
    btn.textContent = 'Stopping...';
    await api('POST', '/api/train/stop');
    pollTrainStatus();
    setTimeout(function() {
      btn.disabled = false;
      btn.textContent = 'Stop';
    }, 1000);
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
}

function syncSlidersFromApplied(params) {
  if (!params || typeof params !== 'object') return;

  // lr
  if ('lr' in params && typeof params.lr === 'number' && params.lr > 0) {
    var lrSlider = $('#knobLr');
    var lrDisplay = $('#knobLrVal');
    if (lrSlider && lrDisplay) {
      lrSlider.value = Math.log10(params.lr);
      lrDisplay.value = params.lr.toExponential(2);
    }
  }

  // weight_decay / wd
  var wd = ('weight_decay' in params) ? params.weight_decay : ('wd' in params ? params.wd : null);
  if (wd !== null && typeof wd === 'number' && wd > 0) {
    var wdSlider = $('#knobWd');
    var wdDisplay = $('#knobWdVal');
    if (wdSlider && wdDisplay) {
      wdSlider.value = Math.log10(wd);
      wdDisplay.value = wd.toExponential(2);
    }
  }

  // weight / loss_w / weight_a (linear 0-1)
  var lw = ('weight' in params) ? params.weight :
           ('loss_w' in params) ? params.loss_w :
           ('weight_a' in params) ? params.weight_a : null;
  if (lw !== null && typeof lw === 'number') {
    var lwSlider = $('#knobLossW');
    var lwDisplay = $('#knobLossWVal');
    if (lwSlider && lwDisplay) {
      lwSlider.value = lw;
      lwDisplay.value = parseFloat(lw).toFixed(2);
    }
  }
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
}

function computeHealth() {
  var lossData = S.metricsData['train_loss'] || S.metricsData['loss'] || [];
  if (lossData.length < 5) { setHealth(50, 'Insufficient data'); return; }
  var recent = lossData.slice(-30);  // larger window for trend
  var first = recent[0].value;
  var last = recent[recent.length - 1].value;
  var trend = (last - first) / Math.max(Math.abs(first), 1e-8);

  var rawScore = 70;
  if (trend < -0.05) rawScore = 90;
  else if (trend < 0) rawScore = 80;
  else if (trend < 0.05) rawScore = 60;
  else if (trend < 0.2) rawScore = 40;
  else rawScore = 20;

  var valLoss = S.metricsData['val_loss'];
  if (valLoss && valLoss.length >= 3) {
    var vRecent = valLoss.slice(-5);
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

async function pollTrainStatus() {
  try {
    var res = await api('GET', '/api/train/status');
    if (!res) return;
    var el = $('#trainStatus');
    var btnStart = $('#btnTrainStart');
    var btnStop = $('#btnTrainStop');
    if (res.running) {
      var info = 'Running since ' + res.started_at;
      if (res.config && res.config.max_steps) {
        info += ' (' + res.config.max_steps + ' steps)';
      }
      if (res.config && res.config.seed !== undefined) {
        info += ' seed=' + res.config.seed;
        // Backfill seed input so user can see/copy it
        var seedInput = document.getElementById('trainSeed');
        if (seedInput && !seedInput.value) seedInput.value = res.config.seed;
      }
      el.textContent = info;
      el.style.color = 'var(--green, #4ade80)';
      btnStart.disabled = true;
      btnStop.disabled = false;
    } else {
      el.textContent = 'Stopped';
      el.style.color = 'var(--text-muted)';
      btnStart.disabled = false;
      btnStop.disabled = true;
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

    if (mode === 'off') {
      statusEl.innerHTML = '<span style="color:var(--text-muted)">Autopilot is off. Select Suggest or Auto to enable.</span>';
      var existing = panel.querySelectorAll('.autopilot-action');
      existing.forEach(function(el) { el.remove(); });
      _lastAutopilotCount = 0;
      return;
    }

    var modeLabel = mode === 'suggest' ? 'Suggesting' : 'Auto-applying';
    statusEl.innerHTML = '<span style="color:var(--cyan)">' + modeLabel + '</span> &middot; ' +
      rulesCount + ' rules &middot; ' + historyCount + ' action(s)';

    if (historyCount !== _lastAutopilotCount) {
      _lastAutopilotCount = historyCount;
      var existing = panel.querySelectorAll('.autopilot-action');
      existing.forEach(function(el) { el.remove(); });
      var toShow = recent.slice().reverse().slice(0, 5);
      toShow.forEach(function(action) {
        var div = document.createElement('div');
        div.className = 'autopilot-action';
        div.style.cssText = 'font-size:10px;padding:4px 0;border-top:1px solid var(--border);margin-top:4px;';
        var statusColor = action.status === 'applied' ? 'var(--green, #4ade80)' :
                          action.status === 'proposed' ? 'var(--yellow, #facc15)' : 'var(--text-muted)';
        var badge = '<span class="ap-badge" style="color:' + statusColor + ';font-weight:600;text-transform:uppercase;font-size:9px">' +
                    action.status + '</span>';
        var step = action.step || '?';
        var ruleId = action.rule_id || '?';
        var condition = action.condition_met || '';
        if (condition.length > 80) condition = condition.substring(0, 77) + '...';
        div.innerHTML = badge + ' <span style="color:var(--text-muted)">step ' + step + '</span> ' +
                        '<span style="color:var(--cyan)">' + ruleId + '</span><br>' +
                        '<span style="color:var(--text-muted)">' + condition + '</span>';

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
