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
