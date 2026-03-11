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
      if (t.dataset.tab === 'recipe-editor') {
        fetchRecipe();
        _startRecipeAutoRefresh();
      } else {
        _stopRecipeAutoRefresh();
      }
      if (t.dataset.tab === 'autopilot-rules') {
        fetchAutopilotRules();
        _startRulesAutoRefresh();
      } else {
        _stopRulesAutoRefresh();
      }
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
/* Recipe Editor — live, editable, auto-refreshing                   */
/* ================================================================ */
var _recipeAutoRefresh = null;
var _recipeEditingIdx = null;  // index being edited, or null

async function fetchRecipe() {
  var data = await api('GET', '/api/recipe/');
  if (!data) return;
  S.recipeEntries = data.entries || [];
  // Don't re-render if user is actively editing an entry
  if (_recipeEditingIdx !== null) return;
  renderRecipe();
}

function _startRecipeAutoRefresh() {
  if (_recipeAutoRefresh) return;
  _recipeAutoRefresh = setInterval(fetchRecipe, 5000);
}
function _stopRecipeAutoRefresh() {
  if (_recipeAutoRefresh) { clearInterval(_recipeAutoRefresh); _recipeAutoRefresh = null; }
}

function renderRecipe() {
  var list = $('#recipeList');
  list.innerHTML = '';
  if (S.recipeEntries.length === 0) {
    list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:11px">No recipe entries. Click + Add to create one, or use Schedule from Controls.</div>';
    return;
  }
  S.recipeEntries.forEach(function(entry, idx) {
    var step = entry.at_step !== undefined ? entry.at_step : (entry.at ? entry.at.step : (entry.step || '?'));
    var mod = entry.module || '?';
    var op = entry.op || '';
    var params = entry.params ? JSON.stringify(entry.params) : '';
    var div = document.createElement('div');
    div.className = 'recipe-entry';
    div.draggable = true;
    div.dataset.idx = idx;
    div.innerHTML =
      '<span class="recipe-step" title="Click to edit step" style="color:var(--text-muted);cursor:pointer">step ' + step + '</span>' +
      '<span class="tl-module ' + mod + '" title="Click to edit"  style="cursor:pointer">' + mod + '</span>' +
      '<span class="recipe-detail" title="Click to edit" style="color:var(--text-secondary);cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + op + ' ' + params + '</span>' +
      '<div style="display:flex;gap:2px;flex-shrink:0">' +
        '<button class="edit-btn" data-idx="' + idx + '" title="Edit" style="border:none;background:none;cursor:pointer;color:var(--text-muted);font-size:11px;padding:0 2px">\u270E</button>' +
        '<button class="del-btn" data-idx="' + idx + '" title="Delete">&times;</button>' +
      '</div>';

    // Drag and drop
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

    // Delete
    div.querySelector('.del-btn').addEventListener('click', async function(e) {
      e.stopPropagation();
      await api('DELETE', '/api/recipe/entry/' + idx);
      fetchRecipe();
    });

    // Inline edit
    div.querySelector('.edit-btn').addEventListener('click', function(e) {
      e.stopPropagation();
      _openInlineEdit(div, entry, idx);
    });

    list.appendChild(div);
  });
}

function _openInlineEdit(div, entry, idx) {
  _recipeEditingIdx = idx;  // Lock auto-refresh from clobbering the form
  var step = entry.at_step !== undefined ? entry.at_step : (entry.at ? entry.at.step : 0);
  var mod = entry.module || 'opt';
  var op = entry.op || 'set_params';
  var params = entry.params ? JSON.stringify(entry.params, null, 0) : '{}';

  div.classList.add('recipe-editing');
  div.draggable = false;
  div.innerHTML =
    '<input type="number" class="re-step" value="' + step + '" style="width:50px;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
    '<select class="re-module" style="width:50px;font-size:10px;padding:2px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px">' +
      '<option value="opt"' + (mod==='opt'?' selected':'') + '>opt</option>' +
      '<option value="loss"' + (mod==='loss'?' selected':'') + '>loss</option>' +
      '<option value="cb"' + (mod==='cb'?' selected':'') + '>cb</option>' +
      '<option value="tune"' + (mod==='tune'?' selected':'') + '>tune</option>' +
    '</select>' +
    '<input type="text" class="re-op" value="' + op + '" placeholder="op" style="width:70px;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
    '<input type="text" class="re-params" value=\'' + params.replace(/'/g, '&#39;') + '\' placeholder=\'{"key":"val"}\' style="flex:1;min-width:60px;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
    '<div style="display:flex;gap:2px;flex-shrink:0">' +
      '<button class="btn btn-sm btn-accent re-save" style="font-size:9px;padding:2px 6px">\u2713</button>' +
      '<button class="btn btn-sm re-cancel" style="font-size:9px;padding:2px 6px">\u2715</button>' +
    '</div>';

  div.querySelector('.re-save').addEventListener('click', async function() {
    var newStep = parseInt(div.querySelector('.re-step').value) || 0;
    var newMod = div.querySelector('.re-module').value;
    var newOp = div.querySelector('.re-op').value.trim() || 'set_params';
    var newParams = {};
    try { newParams = JSON.parse(div.querySelector('.re-params').value || '{}'); } catch(e) {}
    _recipeEditingIdx = null;  // Unlock before fetch
    await api('PUT', '/api/recipe/entry/' + idx, {changes: {
      at_step: newStep, module: newMod, op: newOp, params: newParams
    }});
    fetchRecipe();
  });
  div.querySelector('.re-cancel').addEventListener('click', function() {
    _recipeEditingIdx = null;  // Unlock before fetch
    fetchRecipe();
  });
  // Save on Enter
  div.querySelector('.re-params').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') div.querySelector('.re-save').click();
    if (e.key === 'Escape') div.querySelector('.re-cancel').click();
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
    el.innerHTML = '<div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:4px">TIMELINE PREVIEW</div>';
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
      var data = await api('POST', '/api/recipe/import', {path: path});
      if (data && !data.error) {
        closeModal('modalRecipeImport');
        fetchRecipe();
      } else {
        alert('Import failed: ' + (data ? data.error || data.detail : 'unknown error'));
      }
    } else if (content) {
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
  code += '        step_delay: Delay between steps (seconds). For real training,\n';
  code += '                    set to 0 — the dashboard tracks by step number.\n';
  code += '        stop_event: threading.Event from stdlib. Check stop_event.is_set()\n';
  code += '                    each step to allow graceful shutdown from the dashboard.\n';
  code += '                    No extra imports needed beyond `import threading`.\n';
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
/* Autopilot Rules Editor                                            */
/* ================================================================ */
var _autopilotRules = [];
var _ruleEditingId = null;  // rule_id being edited, or null
var _rulesAutoRefresh = null;

async function fetchAutopilotRules() {
  var data = await api('GET', '/api/autopilot/rules');
  if (!data || !data.rules) return;
  _autopilotRules = data.rules;
  if (_ruleEditingId !== null) return;  // Don't re-render during edit
  renderAutopilotRules();
}

function _startRulesAutoRefresh() {
  if (_rulesAutoRefresh) return;
  _rulesAutoRefresh = setInterval(fetchAutopilotRules, 5000);
}
function _stopRulesAutoRefresh() {
  if (_rulesAutoRefresh) { clearInterval(_rulesAutoRefresh); _rulesAutoRefresh = null; }
}

function renderAutopilotRules() {
  var list = $('#autopilotRulesList');
  if (!list) return;
  list.innerHTML = '';
  if (_autopilotRules.length === 0) {
    list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:11px">No autopilot rules loaded. Click + Add Rule or Reload Defaults.</div>';
    return;
  }
  _autopilotRules.forEach(function(rule) {
    var div = document.createElement('div');
    div.className = 'rule-entry';
    div.dataset.ruleId = rule.rule_id;

    var enabledClass = rule.enabled ? 'rule-enabled' : 'rule-disabled';
    var enabledIcon = rule.enabled ? '\u2713' : '\u2717';
    var enabledColor = rule.enabled ? 'var(--green)' : 'var(--red)';

    var condColor = {plateau: 'var(--yellow)', divergence: 'var(--red)', overfitting: 'var(--orange)', custom: 'var(--purple)'}[rule.condition] || 'var(--text-muted)';
    var confColor = {high: 'var(--green)', medium: 'var(--yellow)', low: 'var(--text-muted)'}[rule.confidence] || 'var(--text-muted)';

    var actionSummary = '';
    if (rule.action) {
      var a = rule.action;
      actionSummary = (a.module || '?') + '.' + (a.op || '?');
      if (a.params) {
        var ps = Object.keys(a.params).slice(0, 3).map(function(k) { return k + '=' + a.params[k]; }).join(', ');
        if (ps) actionSummary += ' ' + ps;
      }
    }

    var paramsSummary = '';
    if (rule.params) {
      paramsSummary = Object.keys(rule.params).map(function(k) { return k + '=' + rule.params[k]; }).join(', ');
    }

    div.innerHTML =
      '<div class="rule-header">' +
        '<button class="rule-toggle" style="border:none;background:none;cursor:pointer;color:' + enabledColor + ';font-size:13px;padding:0 4px" data-rid="' + rule.rule_id + '" title="Toggle enabled">' + enabledIcon + '</button>' +
        '<span class="rule-id" style="font-weight:700;color:var(--text-primary)">' + rule.rule_id + '</span>' +
        '<span style="color:' + condColor + ';font-size:9px;text-transform:uppercase;font-weight:600">' + rule.condition + '</span>' +
        '<span style="color:var(--text-muted);font-size:9px">on <b>' + (rule.metric_name || '?') + '</b></span>' +
        '<span style="color:' + confColor + ';font-size:9px">' + rule.confidence + '</span>' +
        '<div style="flex:1"></div>' +
        '<button class="edit-btn" data-rid="' + rule.rule_id + '" title="Edit">\u270E</button>' +
        '<button class="del-btn" data-rid="' + rule.rule_id + '" title="Delete">&times;</button>' +
      '</div>' +
      (rule.description ? '<div class="rule-desc">' + rule.description + '</div>' : '') +
      '<div class="rule-details">' +
        (paramsSummary ? '<span style="color:var(--text-muted)">if: ' + paramsSummary + '</span>' : '') +
        (actionSummary ? '<span style="color:var(--accent)">then: ' + actionSummary + '</span>' : '') +
      '</div>';

    // Toggle enabled
    div.querySelector('.rule-toggle').addEventListener('click', async function(e) {
      e.stopPropagation();
      await api('POST', '/api/autopilot/rules/' + rule.rule_id + '/toggle');
      fetchAutopilotRules();
    });

    // Delete
    div.querySelector('.del-btn').addEventListener('click', async function(e) {
      e.stopPropagation();
      if (!confirm('Delete rule "' + rule.rule_id + '"?')) return;
      await api('DELETE', '/api/autopilot/rules/' + rule.rule_id);
      fetchAutopilotRules();
    });

    // Edit
    div.querySelector('.edit-btn').addEventListener('click', function(e) {
      e.stopPropagation();
      _openRuleInlineEdit(div, rule);
    });

    list.appendChild(div);
  });
}

function _openRuleInlineEdit(div, rule) {
  _ruleEditingId = rule.rule_id;
  div.className = 'rule-entry rule-editing';

  var params = rule.params ? JSON.stringify(rule.params, null, 0) : '{}';
  var action = rule.action ? JSON.stringify(rule.action, null, 0) : '{}';

  div.innerHTML =
    '<div class="rule-edit-form">' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Condition</label>' +
        '<select class="re-condition" style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px">' +
          '<option value="plateau"' + (rule.condition==='plateau'?' selected':'') + '>plateau</option>' +
          '<option value="divergence"' + (rule.condition==='divergence'?' selected':'') + '>divergence</option>' +
          '<option value="overfitting"' + (rule.condition==='overfitting'?' selected':'') + '>overfitting</option>' +
          '<option value="custom"' + (rule.condition==='custom'?' selected':'') + '>custom</option>' +
        '</select>' +
      '</div>' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Metric</label>' +
        '<input class="re-metric" value="' + (rule.metric_name || '') + '" style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
      '</div>' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Confidence</label>' +
        '<select class="re-confidence" style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px">' +
          '<option value="high"' + (rule.confidence==='high'?' selected':'') + '>high</option>' +
          '<option value="medium"' + (rule.confidence==='medium'?' selected':'') + '>medium</option>' +
          '<option value="low"' + (rule.confidence==='low'?' selected':'') + '>low</option>' +
        '</select>' +
      '</div>' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Params</label>' +
        '<input class="re-params" value=\'' + params.replace(/'/g, '&#39;') + '\' style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
      '</div>' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Action</label>' +
        '<input class="re-action" value=\'' + action.replace(/'/g, '&#39;') + '\' style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px;font-family:var(--font-mono)">' +
      '</div>' +
      '<div style="display:flex;gap:6px;align-items:center">' +
        '<label style="color:var(--text-muted);font-size:9px;width:55px">Desc</label>' +
        '<input class="re-desc" value="' + (rule.description || '').replace(/"/g, '&quot;') + '" style="flex:1;font-size:10px;padding:2px 4px;background:var(--bg-input);color:var(--text-primary);border:1px solid var(--accent);border-radius:3px">' +
      '</div>' +
      '<div style="display:flex;gap:4px;justify-content:flex-end;margin-top:4px">' +
        '<button class="btn btn-sm btn-accent re-save" style="font-size:9px;padding:2px 8px">\u2713 Save</button>' +
        '<button class="btn btn-sm re-cancel" style="font-size:9px;padding:2px 8px">\u2715 Cancel</button>' +
      '</div>' +
    '</div>';

  div.querySelector('.re-save').addEventListener('click', async function() {
    var newCondition = div.querySelector('.re-condition').value;
    var newMetric = div.querySelector('.re-metric').value.trim();
    var newConfidence = div.querySelector('.re-confidence').value;
    var newDesc = div.querySelector('.re-desc').value.trim();
    var newParams = {};
    var newAction = {};
    try { newParams = JSON.parse(div.querySelector('.re-params').value || '{}'); } catch(e) {}
    try { newAction = JSON.parse(div.querySelector('.re-action').value || '{}'); } catch(e) {}
    _ruleEditingId = null;
    await api('PUT', '/api/autopilot/rules/' + rule.rule_id, {
      condition: newCondition,
      metric_name: newMetric,
      confidence: newConfidence,
      params: newParams,
      action: newAction,
      description: newDesc,
    });
    fetchAutopilotRules();
  });

  div.querySelector('.re-cancel').addEventListener('click', function() {
    _ruleEditingId = null;
    fetchAutopilotRules();
  });

  // Enter to save, Escape to cancel (on any input)
  div.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') div.querySelector('.re-save').click();
      if (e.key === 'Escape') div.querySelector('.re-cancel').click();
    });
  });
}

function initAutopilotRulesEditor() {
  var addBtn = $('#btnRuleAdd');
  if (addBtn) addBtn.addEventListener('click', function() { openModal('modalRuleAdd'); });

  var submitBtn = $('#btnRuleSubmit');
  if (submitBtn) submitBtn.addEventListener('click', async function() {
    var ruleId = $('#ruleId').value.trim();
    if (!ruleId) { alert('Rule ID is required'); return; }
    var condition = $('#ruleCondition').value;
    var metric = $('#ruleMetric').value.trim() || 'train_loss';
    var confidence = $('#ruleConfidence').value;
    var params = {};
    var action = {};
    try { params = JSON.parse($('#ruleParams').value || '{}'); } catch(e) {}
    try { action = JSON.parse($('#ruleAction').value || '{}'); } catch(e) {}
    var desc = $('#ruleDescription').value.trim();

    await api('POST', '/api/autopilot/rules', {
      rule_id: ruleId,
      condition: condition,
      metric_name: metric,
      confidence: confidence,
      params: params,
      action: action,
      description: desc,
    });
    closeModal('modalRuleAdd');
    // Clear form
    $('#ruleId').value = '';
    $('#ruleMetric').value = '';
    $('#ruleParams').value = '';
    $('#ruleAction').value = '';
    $('#ruleDescription').value = '';
    fetchAutopilotRules();
  });

  var reloadBtn = $('#btnRulesReload');
  if (reloadBtn) reloadBtn.addEventListener('click', async function() {
    await api('POST', '/api/autopilot/guidelines', {path: '__default__'});
    fetchAutopilotRules();
  });

  fetchAutopilotRules();
}

/* ================================================================ */
/* Compare Runs                                                      */
/* ================================================================ */
var _compareChart = null;
var _compareRunColorPalette = ['#00d4aa', '#3d9eff', '#ff9833', '#ff4d5e', '#9966ff', '#33dd77', '#ff66aa', '#66ddff', '#aadd33', '#dd66ff'];
var _compareRunColorMap = {};  // runId -> color (stable mapping)
var _selectedCompareRuns = new Set();
var _compareAllData = {};  // runId -> records[]
var _compareMetricNames = new Set();
var _compareEnabledMetrics = {};  // name -> bool
var _compareZoomed = false;
var _compareRunMeta = {};  // runId -> run metadata

function _getCompareRunColor(runId) {
  if (!_compareRunColorMap[runId]) {
    var usedColors = Object.values(_compareRunColorMap);
    for (var i = 0; i < _compareRunColorPalette.length; i++) {
      if (usedColors.indexOf(_compareRunColorPalette[i]) === -1) {
        _compareRunColorMap[runId] = _compareRunColorPalette[i];
        return _compareRunColorPalette[i];
      }
    }
    // All colors used, cycle
    _compareRunColorMap[runId] = _compareRunColorPalette[Object.keys(_compareRunColorMap).length % _compareRunColorPalette.length];
  }
  return _compareRunColorMap[runId];
}

function initCompare() {
    var btn = $('#btnCompareRefresh');
    if (btn) btn.addEventListener('click', fetchCompareRuns);

    var zoomBtn = $('#btnCompareZoom');
    if (zoomBtn) zoomBtn.addEventListener('click', function() {
        _compareZoomed = !_compareZoomed;
        document.body.classList.toggle('compare-zoom-mode', _compareZoomed);
        zoomBtn.textContent = _compareZoomed ? '\u2716 Exit Zoom' : '\u26F6 Zoom';
        if (_compareChart) _compareChart.resize();
        _updateCompareOverlayInfo();
    });
}

async function fetchCompareRuns() {
    var data = await api('GET', '/api/train/runs/history');
    if (!data || !data.runs) return;

    // Store run metadata
    data.runs.forEach(function(run) {
        if (run.run_id) _compareRunMeta[run.run_id] = run;
    });

    var list = $('#compareRunList');
    list.innerHTML = '';

    if (data.runs.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted);font-size:10px;padding:8px">No completed runs yet. Start and complete a training run first.</div>';
        return;
    }

    data.runs.forEach(function(run, idx) {
        var div = document.createElement('div');
        var color = _getCompareRunColor(run.run_id);
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

function _updateCompareMetricToggles() {
    var container = $('#compareMetricToggles');
    if (!container) return;
    container.innerHTML = '';
    var names = Array.from(_compareMetricNames).sort();
    if (names.length === 0) return;

    // Build dropdown similar to main metrics dropdown
    var wrap = document.createElement('div');
    wrap.className = 'metric-dropdown-wrap';

    var enabledCount = 0;
    names.forEach(function(n) { if (_compareEnabledMetrics[n] !== false) enabledCount++; });

    var toggleBtn = document.createElement('button');
    toggleBtn.className = 'btn btn-sm metric-dropdown-btn';
    toggleBtn.innerHTML = 'Metrics <span class="metric-count-badge">' + enabledCount + '/' + names.length + '</span> &#9662;';
    toggleBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      var p = wrap.querySelector('.metric-dropdown-panel');
      if (p) p.classList.toggle('open');
    });
    wrap.appendChild(toggleBtn);

    var panel = document.createElement('div');
    panel.className = 'metric-dropdown-panel';

    // Controls
    var controls = document.createElement('div');
    controls.className = 'metric-dropdown-controls';
    var btnAll = document.createElement('button');
    btnAll.className = 'btn btn-sm';
    btnAll.textContent = 'All On';
    btnAll.addEventListener('click', function(e) {
      e.stopPropagation();
      names.forEach(function(n) { _compareEnabledMetrics[n] = true; });
      _rebuildCompareChart(); _updateCompareMetricToggles();
    });
    var btnNone = document.createElement('button');
    btnNone.className = 'btn btn-sm';
    btnNone.textContent = 'All Off';
    btnNone.addEventListener('click', function(e) {
      e.stopPropagation();
      names.forEach(function(n) { _compareEnabledMetrics[n] = false; });
      _rebuildCompareChart(); _updateCompareMetricToggles();
    });
    controls.appendChild(btnAll);
    controls.appendChild(btnNone);
    panel.appendChild(controls);

    var list = document.createElement('div');
    list.className = 'metric-dropdown-list';
    names.forEach(function(name) {
      var row = document.createElement('label');
      row.className = 'metric-dropdown-item';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = _compareEnabledMetrics[name] !== false;
      cb.addEventListener('change', function(e) {
        e.stopPropagation();
        _compareEnabledMetrics[name] = cb.checked;
        var cnt = 0; names.forEach(function(n) { if (_compareEnabledMetrics[n] !== false) cnt++; });
        var badge = wrap.querySelector('.metric-count-badge');
        if (badge) badge.textContent = cnt + '/' + names.length;
        _rebuildCompareChart();
      });
      var label = document.createElement('span');
      label.className = 'metric-dropdown-name';
      label.textContent = name;
      row.appendChild(cb);
      row.appendChild(label);
      list.appendChild(row);
    });
    panel.appendChild(list);
    wrap.appendChild(panel);
    container.appendChild(wrap);

    // Close on outside click
    document.addEventListener('click', function(e) {
      if (!wrap.contains(e.target)) {
        var p = wrap.querySelector('.metric-dropdown-panel');
        if (p) p.classList.remove('open');
      }
    });
}

function _updateCompareOverlayInfo() {
    var overlay = $('#compareOverlayInfo');
    if (!overlay) return;
    if (!_compareZoomed || _selectedCompareRuns.size === 0) {
        overlay.style.display = 'none';
        return;
    }
    overlay.style.display = 'block';
    var html = '';
    var runIds = Array.from(_selectedCompareRuns);
    runIds.forEach(function(runId, idx) {
        var color = _getCompareRunColor(runId);
        var meta = _compareRunMeta[runId] || {};
        html += '<div style="color:' + color + ';margin-bottom:2px">' +
            '<b>' + (meta.config_name || runId) + '</b>' +
            (meta.max_steps ? ' · ' + meta.max_steps + ' steps' : '') +
            '</div>';
    });
    overlay.innerHTML = html;
}

function _rebuildCompareChart() {
    var canvas = document.getElementById('compareChart');
    if (!canvas) return;
    if (_compareChart) _compareChart.destroy();

    var runIds = Array.from(_selectedCompareRuns);
    if (runIds.length === 0) {
        $('#compareSummary').innerHTML = '<div style="color:var(--text-muted);padding:8px">Select runs from the left panel to compare.</div>';
        return;
    }

    var datasets = [];
    var enabledMetrics = [];
    _compareMetricNames.forEach(function(name) {
        if (_compareEnabledMetrics[name] !== false) enabledMetrics.push(name);
    });

    // Mutation annotation plugin for compare chart
    var compareAnnotations = [];

    // Build metric-level dash patterns: first metric solid, rest dashed variants
    var metricDashPatterns = [[], [6, 3], [2, 2], [8, 4, 2, 4], [4, 2], [10, 3]];

    runIds.forEach(function(runId, runIdx) {
        var records = _compareAllData[runId] || [];
        var color = _getCompareRunColor(runId);

        enabledMetrics.forEach(function(metricName, metricIdx) {
            var points = [];
            records.forEach(function(rec) {
                var metrics = rec.metrics || {};
                if (metricName in metrics) {
                    points.push({x: rec.step, y: metrics[metricName]});
                }
            });
            if (points.length === 0) return;

            // All metrics for the same run share the same color, differentiated by dash pattern
            var dashPattern = metricDashPatterns[metricIdx % metricDashPatterns.length];

            datasets.push({
                label: runId.substring(0, 8) + ' · ' + metricName,
                data: points,
                borderColor: color,
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
                borderDash: dashPattern,
                _runId: runId,
                _metricName: metricName,
            });
        });
    });

    // Also fetch applied data per run for mutation markers
    _addCompareMutationMarkers(runIds, datasets, enabledMetrics);

    _compareChart = new Chart(canvas, {
        type: 'line',
        data: {datasets: datasets},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {type: 'linear', title: {display: true, text: 'Step', color: '#7a8fa3', font:{size:11}}, ticks: {color:'#7a8fa3'}, grid: {color: 'rgba(30,46,68,0.5)'}},
                y: {title: {display: false}, ticks: {color:'#7a8fa3'}, grid: {color: 'rgba(30,46,68,0.3)'}},
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#d8e2ec', font: {family: 'JetBrains Mono', size: 10},
                        filter: function(item, chart) {
                            // Hide mutation marker labels from legend
                            if (item.text.indexOf('mutations') !== -1) return false;
                            return true;
                        },
                        usePointStyle: true,
                        pointStyle: 'line',
                    },
                    display: true,
                    position: 'bottom',
                },
                tooltip: {
                    backgroundColor:'#121c2b', borderColor:'#2a4060', borderWidth:1,
                    titleFont:{family:'JetBrains Mono',size:11}, bodyFont:{family:'JetBrains Mono',size:10},
                    callbacks: {
                        label: function(ctx) {
                            var raw = ctx.raw;
                            // Mutation marker tooltip
                            if (raw && raw._mutation) {
                                var m = raw._mutation;
                                var parts = [raw._runId.substring(0, 12) + ' @ step ' + (m.step || raw.x)];
                                parts.push((m.module || '?') + '.' + (m.op || '?'));
                                if (m.params) {
                                    var pStr = Object.keys(m.params).map(function(k) {
                                        var v = m.params[k];
                                        if (typeof v === 'number') v = v < 0.01 ? v.toExponential(1) : parseFloat(v.toPrecision(3));
                                        return k + '=' + v;
                                    }).join(', ');
                                    parts.push(pStr);
                                }
                                var src = m.source || 'interactive';
                                parts.push('source: ' + src);
                                if (raw._metric) parts.push('on: ' + raw._metric);
                                return parts;
                            }
                            // Regular line tooltip — show run color swatch info
                            var ds = ctx.dataset;
                            var val = typeof ctx.parsed.y === 'number' ? ctx.parsed.y.toPrecision(5) : ctx.parsed.y;
                            return ds.label + ': ' + val;
                        }
                    }
                },
            },
            elements: { point: {radius:0, hoverRadius:3}, line: {tension:0.3} },
        },
    });

    _updateCompareOverlayInfo();
    _updateCompareSummary(runIds, enabledMetrics);
}

function _addCompareMutationMarkers(runIds, datasets, enabledMetrics) {
    // For each run, add mutation markers on the first enabled metric that has data
    runIds.forEach(function(runId, runIdx) {
        var color = _getCompareRunColor(runId);
        var appliedData = _compareAllData['_applied_' + runId] || [];
        if (appliedData.length === 0) return;
        var metricRecords = _compareAllData[runId] || [];
        var markerPoints = [];

        appliedData.forEach(function(rec) {
            if (rec.step === undefined) return;
            // Find the closest metric record at or near this step
            var bestIdx = -1;
            var bestDist = Infinity;
            for (var i = 0; i < metricRecords.length; i++) {
                var d = Math.abs(metricRecords[i].step - rec.step);
                if (d < bestDist) { bestDist = d; bestIdx = i; }
                if (metricRecords[i].step >= rec.step) break;
            }
            if (bestIdx < 0) return;
            var m = metricRecords[bestIdx].metrics || {};
            // Use the first enabled metric that has a value at this step
            var y = undefined;
            var usedMetric = null;
            for (var mi = 0; mi < enabledMetrics.length; mi++) {
                if (m[enabledMetrics[mi]] !== undefined) {
                    y = m[enabledMetrics[mi]];
                    usedMetric = enabledMetrics[mi];
                    break;
                }
            }
            if (y !== undefined) {
                markerPoints.push({
                    x: metricRecords[bestIdx].step, y: y,
                    _mutation: rec,
                    _runId: runId,
                    _metric: usedMetric,
                });
            }
        });

        if (markerPoints.length > 0) {
            datasets.push({
                label: runId.substring(0, 8) + ' mutations',
                data: markerPoints,
                type: 'scatter',
                borderColor: color,
                backgroundColor: color + '44',
                pointRadius: 7,
                pointStyle: 'triangle',
                pointHoverRadius: 10,
                showLine: false,
                _isMutationMarker: true,
            });
        }
    });
}

async function _updateCompareSummary(runIds, enabledMetrics) {
    var summary = $('#compareSummary');
    var metricsToShow = enabledMetrics.slice(0, 4);
    var html = '<table style="width:100%;border-collapse:collapse;">';
    html += '<tr style="border-bottom:1px solid var(--border)">';
    html += '<th style="text-align:left;padding:3px 6px;color:var(--text-muted)">Run</th>';
    html += '<th style="text-align:left;padding:3px 6px;color:var(--text-muted)">Config</th>';
    html += '<th style="text-align:right;padding:3px 6px;color:var(--text-muted)">Steps</th>';
    metricsToShow.forEach(function(m) {
        html += '<th style="text-align:right;padding:3px 6px;color:var(--text-muted)">' + m + '</th>';
    });
    html += '</tr>';

    runIds.forEach(function(runId, idx) {
        var color = _getCompareRunColor(runId);
        var meta = _compareRunMeta[runId] || {};
        var records = _compareAllData[runId] || [];
        var applied = _compareAllData['_applied_' + runId] || [];
        var lastRec = records.length > 0 ? records[records.length - 1] : {};
        var lastMetrics = lastRec.metrics || {};

        html += '<tr style="border-bottom:1px solid rgba(30,46,68,0.3)">';
        html += '<td style="padding:3px 6px"><span style="color:' + color + '">' + (runId || '?') + '</span></td>';
        html += '<td style="padding:3px 6px;color:var(--text-secondary)">' + (meta.config_name || meta.config_id || '?') + '</td>';
        html += '<td style="padding:3px 6px;text-align:right">' + (meta.max_steps || records.length || '?') + '</td>';
        metricsToShow.forEach(function(m) {
            var v = lastMetrics[m];
            var display = v !== undefined ? (typeof v === 'number' ? v.toFixed(6) : v) : '--';
            html += '<td style="padding:3px 6px;text-align:right;font-weight:600">' + display + '</td>';
        });
        html += '</tr>';

        // Show mutation details for this run
        if (applied.length > 0) {
            html += '<tr><td colspan="' + (3 + metricsToShow.length) + '" style="padding:2px 6px 6px 20px;font-size:9px;">';
            html += '<span style="color:var(--orange);font-weight:700">' + applied.length + ' mutations:</span> ';
            applied.slice(0, 8).forEach(function(a, ai) {
                var src = a.source || 'interactive';
                var srcColor = src === 'recipe' ? 'var(--yellow)' : src === 'autopilot' ? 'var(--cyan)' : 'var(--text-muted)';
                var paramStr = '';
                if (a.params) {
                    paramStr = Object.keys(a.params).slice(0, 2).map(function(k) {
                        var v = a.params[k];
                        if (typeof v === 'number') v = v < 0.01 ? v.toExponential(1) : parseFloat(v.toPrecision(3));
                        return k + '\u2192' + v;
                    }).join(', ');
                }
                if (ai > 0) html += ' ';
                html += '<span style="color:var(--text-secondary)">[s' + (a.step||'?') + ' ' + (a.module||'?') + '.' + (a.op||'?');
                if (paramStr) html += ' ' + paramStr;
                html += ' <span style="color:' + srcColor + '">' + src + '</span>]</span>';
            });
            if (applied.length > 8) html += ' <span style="color:var(--text-muted)">+' + (applied.length - 8) + ' more</span>';
            html += '</td></tr>';
        }
    });
    html += '</table>';
    summary.innerHTML = html;
}

async function updateCompareChart() {
    if (_selectedCompareRuns.size === 0) {
        if (_compareChart) {
            _compareChart.destroy();
            _compareChart = null;
        }
        _compareMetricNames = new Set();
        _updateCompareMetricToggles();
        $('#compareSummary').innerHTML = '<div style="color:var(--text-muted);padding:8px">Select runs from the left panel to compare.</div>';
        return;
    }

    // Fetch metrics and applied data for each selected run
    var promises = [];
    var runIds = Array.from(_selectedCompareRuns);

    for (var i = 0; i < runIds.length; i++) {
        (function(runId) {
            // Fetch metrics
            var p1 = api('GET', '/api/train/runs/' + runId + '/metrics').then(function(data) {
                if (data && data.records) {
                    _compareAllData[runId] = data.records;
                    // Discover metric names
                    data.records.forEach(function(rec) {
                        var metrics = rec.metrics || {};
                        Object.keys(metrics).forEach(function(name) {
                            if (typeof metrics[name] === 'number') {
                                _compareMetricNames.add(name);
                                if (!(name in _compareEnabledMetrics)) {
                                    // Default: enable all discovered metrics
                                    _compareEnabledMetrics[name] = true;
                                }
                            }
                        });
                    });
                }
            });
            // Fetch applied data for mutation markers
            var p2 = api('GET', '/api/train/runs/' + runId + '/applied').then(function(data) {
                if (data && data.records) {
                    _compareAllData['_applied_' + runId] = data.records;
                }
            }).catch(function() {
                _compareAllData['_applied_' + runId] = [];
            });
            promises.push(p1, p2);
        })(runIds[i]);
    }

    await Promise.all(promises);

    // Ensure at least one metric is enabled
    var anyEnabled = false;
    _compareMetricNames.forEach(function(name) {
        if (_compareEnabledMetrics[name]) anyEnabled = true;
    });
    if (!anyEnabled) {
        // Enable train_loss by default, or first metric
        if (_compareMetricNames.has('train_loss')) {
            _compareEnabledMetrics['train_loss'] = true;
        } else {
            var first = null;
            _compareMetricNames.forEach(function(n) { if (!first) first = n; });
            if (first) _compareEnabledMetrics[first] = true;
        }
    }

    _updateCompareMetricToggles();
    _rebuildCompareChart();
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
