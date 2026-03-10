/**
 * hotcb dashboard — recipe editor, chat, notifications, timeline, call-for-help
 */

/* ================================================================ */
/* Tabs                                                              */
/* ================================================================ */
var _manifoldRefreshInterval = null;

function _startManifoldAutoRefresh() {
  if (_manifoldRefreshInterval) return;
  _manifoldRefreshInterval = setInterval(function() {
    var activeSubtab = document.querySelector('[data-subtab].active');
    if (activeSubtab && activeSubtab.dataset.subtab === 'feature-space') {
      fetchFeatures();
    } else {
      fetchManifold();
    }
  }, 10000);
}

function _stopManifoldAutoRefresh() {
  if (_manifoldRefreshInterval) {
    clearInterval(_manifoldRefreshInterval);
    _manifoldRefreshInterval = null;
  }
}

function initTabs() {
  $$('.tabs .tab[data-tab]').forEach(function(t) {
    t.addEventListener('click', function() {
      var tabGroup = t.closest('.tabs');
      var area = tabGroup.parentElement;
      tabGroup.querySelectorAll('.tab[data-tab]').forEach(function(x) { x.classList.remove('active'); });
      t.classList.add('active');
      area.querySelectorAll('.tab-content[data-tab]').forEach(function(x) { x.classList.remove('active'); });
      var target = area.querySelector('.tab-content[data-tab="' + t.dataset.tab + '"]');
      if (target) target.classList.add('active');
      if (t.dataset.tab === 'manifold') {
        fetchManifold();
        _startManifoldAutoRefresh();
      } else {
        _stopManifoldAutoRefresh();
      }
      if (t.dataset.tab === 'features') fetchFeatures();
      if (t.dataset.tab === 'recipe-editor') fetchRecipe();
      if (t.dataset.tab === 'compare') fetchCompareRuns();
    });
  });

  $$('[data-subtab]').forEach(function(t) {
    t.addEventListener('click', function() {
      $$('[data-subtab]').forEach(function(x) { x.classList.remove('active'); });
      t.classList.add('active');
      if (t.dataset.subtab === 'metric-space') fetchManifold();
      else fetchFeatures();
    });
  });
}

/* ================================================================ */
/* Timeline                                                          */
/* ================================================================ */
function clearTimelineDedup() {
  S._timelineKeys = {};
}

function addTimelineItem(rec) {
  var list = $('#timelineList');

  // Deduplicate: skip if an item with the same step+module+op already exists
  var dedupKey = (rec.step || '?') + ':' + (rec.module || '?') + ':' + (rec.op || '');
  if (!S._timelineKeys) S._timelineKeys = {};
  if (S._timelineKeys[dedupKey]) return;
  S._timelineKeys[dedupKey] = true;

  var div = document.createElement('div');
  div.className = 'timeline-item';
  div.setAttribute('data-dedup-key', dedupKey);
  div.style.cursor = 'pointer';
  var step = rec.step || '?';
  var mod = rec.module || '?';
  var desc = rec.op || '';
  var params = rec.params ? JSON.stringify(rec.params) : '';
  var decision = rec.decision || rec.status || 'applied';
  var source = rec.source || 'interactive';
  var sourceColor = source === 'recipe' ? 'var(--yellow, #facc15)' :
                    source === 'autopilot' ? 'var(--cyan, #22d3ee)' :
                    'var(--text-muted)';
  div.innerHTML =
    '<span class="tl-step">step ' + step + '</span>' +
    '<span class="tl-module ' + mod + '">' + mod + '</span>' +
    '<span style="color:var(--text-secondary)">' + desc + ' ' + params + '</span>' +
    '<span class="tl-decision ' + decision + '">' + decision + '</span>' +
    '<span style="color:' + sourceColor + ';font-size:9px;margin-left:4px;text-transform:uppercase;font-weight:600">' + source + '</span>';

  // Click handler: highlight annotation on chart and show impact summary
  div.addEventListener('click', function() {
    // Remove active state from all timeline items
    list.querySelectorAll('.timeline-item').forEach(function(item) {
      item.classList.remove('tl-active');
    });

    // Toggle: if clicking the same item, deselect
    if (_highlightedMutationStep === rec.step) {
      _highlightedMutationStep = null;
      div.classList.remove('tl-active');
      var existing = document.getElementById('impactSummary');
      if (existing) existing.remove();
      if (S.chartInstance) S.chartInstance.update('none');
      return;
    }

    // Highlight this item
    div.classList.add('tl-active');
    _highlightedMutationStep = rec.step;

    // Scroll chart to center on the mutation step
    scrollChartToStep(rec.step);

    // Redraw chart to update annotation highlight
    if (S.chartInstance) S.chartInstance.update('none');

    // Show impact summary panel
    renderImpactSummary(rec);
  });

  list.prepend(div);
  $('#mutationCount').textContent = S.appliedData.length;
}

/* ================================================================ */
/* Recipe Editor                                                     */
/* ================================================================ */
async function fetchRecipe() {
  var data = await api('GET', '/api/recipe/');
  if (!data || !data.entries) return;
  S.recipeEntries = data.entries;
  renderRecipe();
}

function renderRecipe() {
  var list = $('#recipeList');
  list.innerHTML = '';
  if (S.recipeEntries.length === 0) {
    list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:11px">No recipe entries. Click + Add to create one.</div>';
    return;
  }
  S.recipeEntries.forEach(function(entry, idx) {
    var step = entry.at ? entry.at.step : (entry.step || '?');
    var div = document.createElement('div');
    div.className = 'recipe-entry';
    div.draggable = true;
    div.dataset.idx = idx;
    var params = entry.params ? JSON.stringify(entry.params) : '';
    div.innerHTML =
      '<span style="color:var(--text-muted)">step ' + step + '</span>' +
      '<span class="tl-module ' + (entry.module || '') + '">' + (entry.module || '?') + '</span>' +
      '<span style="color:var(--text-secondary)">' + (entry.op || '') + ' ' + params + '</span>' +
      '<button class="del-btn" data-idx="' + idx + '">&times;</button>';
    div.addEventListener('dragstart', function(e) { e.dataTransfer.setData('text/plain', idx); div.classList.add('dragging'); });
    div.addEventListener('dragend', function() { div.classList.remove('dragging'); });
    div.addEventListener('dragover', function(e) { e.preventDefault(); });
    div.addEventListener('drop', async function(e) {
      e.preventDefault();
      var fromIdx = parseInt(e.dataTransfer.getData('text/plain'));
      if (fromIdx === idx) return;
      await api('POST', '/api/recipe/move', {from: fromIdx, to: idx});
      fetchRecipe();
    });
    div.querySelector('.del-btn').addEventListener('click', async function() {
      await api('DELETE', '/api/recipe/entry/' + idx);
      fetchRecipe();
    });
    list.appendChild(div);
  });
}

function initRecipeEditor() {
  $('#btnRecipeAdd').addEventListener('click', function() { openModal('modalRecipeAdd'); });

  $('#btnRecipeSubmit').addEventListener('click', async function() {
    var step = parseInt($('#recipeStep').value);
    var module = $('#recipeModule').value;
    var op = $('#recipeOp').value;
    var params = {};
    try { params = JSON.parse($('#recipeParams').value || '{}'); } catch(e) {}
    await api('POST', '/api/recipe/entry', {entry: {at: {step: step}, module: module, op: op, params: params}});
    closeModal('modalRecipeAdd');
    fetchRecipe();
  });

  $('#btnRecipeExport').addEventListener('click', async function() {
    var data = await api('POST', '/api/recipe/export', {path: 'exported_recipe.jsonl'});
    if (data) {
      alert('Recipe exported to: ' + (data.path || 'exported_recipe.jsonl'));
    }
  });

  $('#btnRecipeValidate').addEventListener('click', async function() {
    var data = await api('GET', '/api/recipe/validate');
    if (data) alert(data.valid ? 'Recipe is valid!' : 'Errors: ' + JSON.stringify(data.errors));
  });

  $('#btnRecipeReplay').addEventListener('click', async function() {
    var el = $('#recipeReplayPreview');
    var data = await api('GET', '/api/recipe/timeline');
    if (!data || !data.timeline) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:4px">REPLAY PREVIEW</div>';
    data.timeline.forEach(function(t) {
      var d = document.createElement('div');
      d.className = 'timeline-item';
      d.innerHTML = '<span class="tl-step">step ' + (t.step || '?') + '</span>' +
        '<span class="tl-module ' + (t.module || '') + '">' + (t.module || '?') + '</span>' +
        '<span>' + (t.op || '') + ' ' + JSON.stringify(t.params || {}) + '</span>' +
        '<span style="color:var(--text-muted)">&rarr;</span>';
      el.appendChild(d);
    });
  });

  $('#btnRecipeImport').addEventListener('click', function() { openModal('modalRecipeImport'); });

  $('#btnRecipeImportSubmit').addEventListener('click', async function() {
    var content = $('#recipeImportContent').value.trim();
    var path = $('#recipeImportPath').value.trim();

    if (path) {
      // Import from server file path
      var data = await api('POST', '/api/recipe/import', {path: path});
      if (data && !data.error) {
        closeModal('modalRecipeImport');
        fetchRecipe();
      } else {
        alert('Import failed: ' + (data ? data.error || data.detail : 'unknown error'));
      }
    } else if (content) {
      // Parse JSONL content and add entries one by one
      var lines = content.split('\n').filter(function(l) { return l.trim(); });
      var added = 0;
      for (var i = 0; i < lines.length; i++) {
        try {
          var entry = JSON.parse(lines[i]);
          await api('POST', '/api/recipe/entry', {entry: entry});
          added++;
        } catch(e) {
          console.warn('Skipping invalid line:', lines[i]);
        }
      }
      closeModal('modalRecipeImport');
      fetchRecipe();
      if (added > 0) alert('Imported ' + added + ' entries');
    } else {
      alert('Please provide content or a file path');
    }
  });

  $('#btnRecipeDiff').addEventListener('click', async function() {
    var el = $('#recipeDiffView');
    var data = await api('POST', '/api/recipe/diff', {other_path: 'hotcb.recipe.jsonl'});
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:4px">DIFF VIEW</div>';
    if (data && data.diffs) {
      data.diffs.forEach(function(d) {
        var line = document.createElement('div');
        var type = d.type || 'same';
        line.className = 'diff-line ' + type;
        line.textContent = (type === 'added' ? '+' : type === 'removed' ? '-' : '~') + ' ' + JSON.stringify(d.entry || d.before || d);
        el.appendChild(line);
      });
    } else {
      S.recipeEntries.forEach(function(e) {
        var line = document.createElement('div');
        line.className = 'diff-line remove';
        line.textContent = '- ' + JSON.stringify(e);
        el.appendChild(line);
      });
    }
  });
}

/* ================================================================ */
/* Chat (NL / vibe-coder)                                            */
/* ================================================================ */
function initChat() {
  $('#btnChatSend').addEventListener('click', sendChat);
  $('#chatInput').addEventListener('keydown', function(e) { if (e.key === 'Enter') sendChat(); });
}

async function sendChat() {
  var input = $('#chatInput');
  var msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addChatMsg('user', msg);
  var data = await api('POST', '/api/chat', {message: msg});
  if (data) {
    addChatMsg('bot', data.reply);
    if (data.command) addChatMsg('bot', 'Queued: ' + JSON.stringify(data.command));
  } else {
    addChatMsg('bot', 'Failed to process command.');
  }
}

function addChatMsg(role, text) {
  var container = $('#chatMessages');
  var div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  S.chatHistory.push({role: role, text: text});
}

/* ================================================================ */
/* Notifications / Alerts                                            */
/* ================================================================ */
function initNotifications() {
  $('#btnNotif').addEventListener('click', function() {
    var card = $('#notifCard');
    card.style.display = card.style.display === 'none' ? 'block' : 'none';
    if (card.style.display === 'block') fetchAlerts();
  });

  $('#btnAddAlert').addEventListener('click', function() { openModal('modalAddAlert'); });

  $('#btnAlertSubmit').addEventListener('click', async function() {
    var rule = {
      metric_name: $('#alertMetric').value,
      condition: $('#alertCondition').value,
      threshold: parseFloat($('#alertThreshold').value),
      channel: $('#alertChannel').value,
      message: 'Alert: ' + $('#alertMetric').value + ' ' + $('#alertCondition').value + ' ' + $('#alertThreshold').value,
    };
    await api('POST', '/api/notifications/rules', rule);
    closeModal('modalAddAlert');
    fetchAlerts();
  });
}

async function fetchAlerts() {
  var data = await api('GET', '/api/notifications/alerts');
  var container = $('#alertList');
  if (!data || !data.alerts || data.alerts.length === 0) {
    container.innerHTML = '<div style="font-size:10px;color:var(--text-muted)">No alerts fired</div>';
    return;
  }
  container.innerHTML = '';
  data.alerts.slice(-20).forEach(function(a) {
    var div = document.createElement('div');
    div.className = 'notif-item ' + (a.severity || 'info');
    div.textContent = '[' + (a.metric || '?') + '] ' + (a.message || a.condition || '');
    container.appendChild(div);
  });
}

/* ================================================================ */
/* Call for Help                                                     */
/* ================================================================ */
function initCallHelp() {
  $('#btnCallHelp').addEventListener('click', function() {
    var lines = ['=== hotcb Training Snapshot ===', ''];
    lines.push('Timestamp: ' + new Date().toISOString());
    lines.push('');
    lines.push('--- Latest Metrics ---');
    Object.keys(S.latestMetrics).forEach(function(k) {
      lines.push('  ' + k + ': ' + fmtNum(S.latestMetrics[k]));
    });
    lines.push('');
    lines.push('--- Recent Interventions ---');
    S.appliedData.slice(-5).forEach(function(a) {
      lines.push('  step ' + a.step + ': ' + a.module + '.' + a.op + ' ' + JSON.stringify(a.params || {}));
    });
    lines.push('');
    lines.push('--- Active Alerts ---');
    if (S.alerts.length === 0) lines.push('  (none)');
    else S.alerts.slice(-5).forEach(function(a) {
      lines.push('  [' + (a.severity || 'info') + '] ' + (a.message || ''));
    });
    lines.push('');
    lines.push('---');
    lines.push('Sent from hotcb dashboard');
    $('#helpMessage').value = lines.join('\n');
    openModal('modalCallHelp');
  });

  $('#btnCopyHelp').addEventListener('click', function() {
    navigator.clipboard.writeText($('#helpMessage').value);
    $('#btnCopyHelp').textContent = 'Copied!';
    setTimeout(function() { $('#btnCopyHelp').textContent = 'Copy to Clipboard'; }, 1500);
  });
}

/* ================================================================ */
/* Config Wizard                                                     */
/* ================================================================ */
function initConfigWizard() {
  var btnPreview = $('#btnConfigPreview');
  var btnExport = $('#btnConfigExport');
  var btnRegister = $('#btnConfigRegister');

  if (btnPreview) btnPreview.addEventListener('click', function() {
    var code = _generateConfigCode();
    var preview = $('#cfgPreview');
    var codeEl = $('#cfgPreviewCode');
    if (preview && codeEl) {
      codeEl.textContent = code;
      preview.style.display = 'block';
    }
  });

  if (btnExport) btnExport.addEventListener('click', function() {
    var code = _generateConfigCode();
    var blob = new Blob([code], {type: 'text/plain'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    var cfgId = ($('#cfgId').value || 'config').trim().replace(/\s+/g, '_');
    a.download = cfgId + '_config.py';
    a.click();
    URL.revokeObjectURL(url);
  });

  if (btnRegister) btnRegister.addEventListener('click', async function() {
    var cfgId = ($('#cfgId').value || '').trim();
    var cfgName = ($('#cfgName').value || '').trim();
    var trainFn = ($('#cfgTrainFn').value || '').trim();
    if (!cfgId || !cfgName || !trainFn) {
      alert('Config ID, Name, and Training Function Path are required.');
      return;
    }
    var body = {
      config_id: cfgId,
      name: cfgName,
      description: ($('#cfgDesc').value || '').trim(),
      train_fn_path: trainFn,
      defaults: {
        max_steps: parseInt($('#cfgMaxSteps').value) || 1000,
        step_delay: parseFloat($('#cfgStepDelay').value) || 0.1,
      },
      recipe_path: ($('#cfgRecipePath').value || '').trim() || null,
    };
    btnRegister.disabled = true;
    btnRegister.textContent = 'Registering...';
    var res = await api('POST', '/api/train/configs/register', body);
    btnRegister.disabled = false;
    btnRegister.textContent = 'Register';
    if (res && res.status === 'registered') {
      alert('Config "' + cfgId + '" registered! It will appear in the Training dropdown.');
      // Refresh the training config dropdown
      var configRes = await api('GET', '/api/train/configs');
      if (configRes && configRes.configs) {
        var sel = $('#trainConfig');
        sel.innerHTML = '';
        configRes.configs.forEach(function(cfg) {
          var opt = document.createElement('option');
          opt.value = cfg.config_id;
          opt.textContent = cfg.name;
          sel.appendChild(opt);
        });
        sel.value = cfgId;
        sel.dispatchEvent(new Event('change'));
      }
    } else {
      alert('Registration failed: ' + (res ? res.error || JSON.stringify(res) : 'unknown error'));
    }
  });
}

function _generateConfigCode() {
  var cfgId = ($('#cfgId').value || 'my_config').trim();
  var cfgName = ($('#cfgName').value || 'My Config').trim();
  var cfgDesc = ($('#cfgDesc').value || '').trim();
  var maxSteps = $('#cfgMaxSteps').value || '1000';
  var stepDelay = $('#cfgStepDelay').value || '0.1';
  var trainFn = ($('#cfgTrainFn').value || 'my_module:my_training_fn').trim();
  var recipePath = ($('#cfgRecipePath').value || '').trim();

  var code = '"""Custom training configuration for hotcb."""\n';
  code += 'import json\n';
  code += 'import os\n';
  code += 'import time\n';
  code += 'import threading\n\n';
  code += 'from hotcb.server.launcher import TrainingLauncher, TrainingConfig\n\n\n';
  code += 'def ' + cfgId + '_training(run_dir, max_steps, step_delay, stop_event):\n';
  code += '    """' + (cfgDesc || 'Custom training loop.') + '\n\n';
  code += '    Args:\n';
  code += '        run_dir: Directory for JSONL I/O files\n';
  code += '        max_steps: Maximum training steps\n';
  code += '        step_delay: Delay between steps (seconds)\n';
  code += '        stop_event: threading.Event — check .is_set() each step\n';
  code += '    """\n';
  code += '    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")\n';
  code += '    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")\n';
  code += '    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")\n';
  code += '    cmd_offset = 0\n\n';
  code += '    # TODO: Initialize your model, optimizer, loss, dataset here\n';
  code += '    # model = ...\n';
  code += '    # optimizer = ...\n\n';
  code += '    for step in range(1, max_steps + 1):\n';
  code += '        if stop_event.is_set():\n';
  code += '            break\n\n';
  code += '        # Read commands from dashboard\n';
  code += '        # cmds, cmd_offset = _read_commands(commands_path, cmd_offset)\n';
  code += '        # for cmd in cmds: apply_command(cmd)\n\n';
  code += '        # TODO: Your training step here\n';
  code += '        loss = 1.0  # Replace with actual loss\n\n';
  code += '        # Write metrics\n';
  code += '        with open(metrics_path, "a") as f:\n';
  code += '            f.write(json.dumps({"step": step, "metrics": {\n';
  code += '                "train_loss": loss,\n';
  code += '                # Add your metrics here\n';
  code += '            }}) + "\\n")\n\n';
  code += '        if step_delay > 0:\n';
  code += '            time.sleep(step_delay)\n\n\n';
  code += '# Register with the launcher\n';
  code += 'config = TrainingConfig(\n';
  code += '    config_id="' + cfgId + '",\n';
  code += '    name="' + cfgName + '",\n';
  code += '    description="' + cfgDesc.replace(/"/g, '\\"') + '",\n';
  code += '    train_fn=' + cfgId + '_training,\n';
  code += '    defaults={"max_steps": ' + maxSteps + ', "step_delay": ' + stepDelay + '},\n';
  code += ')\n';
  if (recipePath) {
    code += '\n# Pre-load recipe\n';
    code += '# launcher.register_config(config)\n';
    code += '# Recipe: ' + recipePath + '\n';
  }
  return code;
}

/* ================================================================ */
/* Compare Runs                                                      */
/* ================================================================ */
var _compareChart = null;
var _compareRunColors = ['#00d4aa', '#3d9eff', '#ff9833', '#ff4d5e', '#9966ff', '#33dd77'];
var _selectedCompareRuns = new Set();

function initCompare() {
    var btn = $('#btnCompareRefresh');
    if (btn) btn.addEventListener('click', fetchCompareRuns);
}

async function fetchCompareRuns() {
    var data = await api('GET', '/api/train/runs/history');
    if (!data || !data.runs) return;

    var list = $('#compareRunList');
    list.innerHTML = '';

    if (data.runs.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted);font-size:10px;padding:8px">No completed runs yet. Start and complete a training run first.</div>';
        return;
    }

    data.runs.forEach(function(run, idx) {
        var div = document.createElement('div');
        var color = _compareRunColors[idx % _compareRunColors.length];
        var isSelected = _selectedCompareRuns.has(run.run_id);
        div.style.cssText = 'display:flex;align-items:center;gap:6px;padding:6px 8px;border:1px solid ' +
            (isSelected ? color : 'var(--border)') + ';border-radius:4px;cursor:pointer;font-size:10px;' +
            'background:' + (isSelected ? color + '11' : 'transparent') + ';transition:all 0.15s;';

        var dot = '<span style="width:8px;height:8px;border-radius:50%;background:' + color + ';flex-shrink:0;display:inline-block"></span>';
        var configLabel = run.config_name || run.config_id || '?';
        var runId = run.run_id || '?';
        var finalLoss = run.final_metrics && run.final_metrics.train_loss
            ? run.final_metrics.train_loss.toFixed(4) : '--';

        div.innerHTML = dot +
            '<div style="flex:1;overflow:hidden">' +
            '<div style="font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + configLabel + '</div>' +
            '<div style="color:var(--text-muted);font-size:9px">' + runId + ' · loss: ' + finalLoss + '</div>' +
            '</div>';

        div.addEventListener('click', function() {
            if (_selectedCompareRuns.has(run.run_id)) {
                _selectedCompareRuns.delete(run.run_id);
                div.style.borderColor = 'var(--border)';
                div.style.background = 'transparent';
            } else {
                _selectedCompareRuns.add(run.run_id);
                div.style.borderColor = color;
                div.style.background = color + '11';
            }
            updateCompareChart();
        });

        list.appendChild(div);
    });
}

async function updateCompareChart() {
    if (_selectedCompareRuns.size === 0) {
        if (_compareChart) {
            _compareChart.destroy();
            _compareChart = null;
        }
        $('#compareSummary').innerHTML = '<div style="color:var(--text-muted);padding:8px">Select runs from the left panel to compare.</div>';
        return;
    }

    // Fetch metrics for each selected run
    var allData = {};
    var promises = [];
    var runIds = Array.from(_selectedCompareRuns);

    for (var i = 0; i < runIds.length; i++) {
        (function(runId, idx) {
            var p = api('GET', '/api/train/runs/' + runId + '/metrics').then(function(data) {
                if (data && data.records) {
                    allData[runId] = data.records;
                }
            });
            promises.push(p);
        })(runIds[i], i);
    }

    await Promise.all(promises);

    // Build chart datasets
    var canvas = document.getElementById('compareChart');
    if (!canvas) return;

    if (_compareChart) _compareChart.destroy();

    var datasets = [];
    var metricName = 'train_loss'; // Default comparison metric

    runIds.forEach(function(runId, idx) {
        var records = allData[runId] || [];
        var color = _compareRunColors[idx % _compareRunColors.length];
        var points = [];
        records.forEach(function(rec) {
            var metrics = rec.metrics || {};
            if (metricName in metrics) {
                points.push({x: rec.step, y: metrics[metricName]});
            }
        });
        datasets.push({
            label: runId,
            data: points,
            borderColor: color,
            backgroundColor: color + '22',
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
        });
    });

    _compareChart = new Chart(canvas, {
        type: 'line',
        data: {datasets: datasets},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {type: 'linear', title: {display: true, text: 'Step', color: '#7a8fa3'}, grid: {color: '#1e2e44'}},
                y: {title: {display: true, text: metricName, color: '#7a8fa3'}, grid: {color: '#1e2e44'}},
            },
            plugins: {
                legend: {labels: {color: '#d8e2ec', font: {family: 'JetBrains Mono', size: 10}}},
            },
        },
    });

    // Build summary table
    var summary = $('#compareSummary');
    var html = '<table style="width:100%;border-collapse:collapse;">';
    html += '<tr style="border-bottom:1px solid var(--border)"><th style="text-align:left;padding:3px 6px;color:var(--text-muted)">Run</th>';
    html += '<th style="text-align:left;padding:3px 6px;color:var(--text-muted)">Config</th>';
    html += '<th style="text-align:right;padding:3px 6px;color:var(--text-muted)">Steps</th>';
    html += '<th style="text-align:right;padding:3px 6px;color:var(--text-muted)">Final Loss</th></tr>';

    // Fetch run metadata for the summary
    var histData = await api('GET', '/api/train/runs/history');
    if (histData && histData.runs) {
        histData.runs.forEach(function(run, idx) {
            if (!_selectedCompareRuns.has(run.run_id)) return;
            var color = _compareRunColors[runIds.indexOf(run.run_id) % _compareRunColors.length];
            var finalLoss = run.final_metrics && run.final_metrics.train_loss
                ? run.final_metrics.train_loss.toFixed(6) : '--';
            html += '<tr style="border-bottom:1px solid rgba(30,46,68,0.3)">';
            html += '<td style="padding:3px 6px"><span style="color:' + color + '">' + (run.run_id || '?') + '</span></td>';
            html += '<td style="padding:3px 6px;color:var(--text-secondary)">' + (run.config_name || run.config_id || '?') + '</td>';
            html += '<td style="padding:3px 6px;text-align:right">' + (run.max_steps || '?') + '</td>';
            html += '<td style="padding:3px 6px;text-align:right;font-weight:600">' + finalLoss + '</td>';
            html += '</tr>';
        });
    }
    html += '</table>';
    summary.innerHTML = html;
}

/* ================================================================ */
/* Multi-run selector                                                */
/* ================================================================ */
function updateRunSelector(allDirs) {
  var container = $('#runSelector');
  container.innerHTML = '';
  if (!allDirs || allDirs.length <= 1) return;
  allDirs.forEach(function(dir, i) {
    var chip = document.createElement('span');
    chip.className = 'run-chip' + (S.activeRuns.has(i) ? ' active' : '');
    chip.textContent = 'Run ' + i;
    chip.title = dir;
    chip.addEventListener('click', function() {
      if (S.activeRuns.has(i)) S.activeRuns.delete(i);
      else S.activeRuns.add(i);
      chip.classList.toggle('active');
      updateChart();
    });
    container.appendChild(chip);
  });
}
