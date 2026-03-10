/**
 * hotcb dashboard — Unified chart system with per-metric cards
 */

// Forecast update interval
var _forecastTimer = null;
var _forecastCache = {};  // metric -> {forecast, mutation}

// Highlighted mutation step (set when user clicks a timeline item)
var _highlightedMutationStep = null;

// ---- Mutation annotation plugin for Chart.js ----
var mutationAnnotationPlugin = {
  id: 'mutationAnnotations',
  afterDraw: function(chart) {
    if (!S.appliedData || S.appliedData.length === 0) return;
    var xScale = chart.scales.x;
    var yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    var ctx = chart.ctx;
    var top = yScale.top;
    var bottom = yScale.bottom;

    S.appliedData.forEach(function(rec) {
      var step = rec.step;
      if (step === undefined || step === null) return;
      if (step < xScale.min || step > xScale.max) return;
      var x = xScale.getPixelForValue(step);

      var isHighlighted = (_highlightedMutationStep === step);

      // Draw vertical dashed line
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash(isHighlighted ? [6, 3] : [4, 4]);
      ctx.lineWidth = isHighlighted ? 2 : 1;
      ctx.strokeStyle = isHighlighted
        ? 'rgba(255, 152, 51, 0.85)'
        : 'rgba(255, 152, 51, 0.35)';
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.restore();

      // Build compact label from params
      var label = '';
      if (rec.params && typeof rec.params === 'object') {
        var keys = Object.keys(rec.params);
        var parts = [];
        keys.slice(0, 2).forEach(function(k) {
          var v = rec.params[k];
          if (typeof v === 'number') {
            v = v < 0.01 || v > 1e4 ? v.toExponential(1) : parseFloat(v.toPrecision(3));
          }
          parts.push(k + '\u2192' + v);
        });
        label = parts.join(', ');
      } else if (rec.op) {
        label = rec.op;
      }

      if (label) {
        ctx.save();
        ctx.font = (isHighlighted ? 'bold ' : '') + '9px "JetBrains Mono", monospace';
        var textWidth = ctx.measureText(label).width;
        var pad = 3;
        var labelX = x - textWidth / 2 - pad;
        var labelY = top - 2;

        // Background pill
        ctx.fillStyle = isHighlighted
          ? 'rgba(255, 152, 51, 0.25)'
          : 'rgba(255, 152, 51, 0.12)';
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(labelX, labelY - 11, textWidth + pad * 2, 13, 3);
        } else {
          ctx.rect(labelX, labelY - 11, textWidth + pad * 2, 13);
        }
        ctx.fill();

        // Text
        ctx.fillStyle = isHighlighted
          ? 'rgba(255, 200, 120, 1)'
          : 'rgba(255, 200, 120, 0.7)';
        ctx.textBaseline = 'bottom';
        ctx.fillText(label, labelX + pad, labelY);
        ctx.restore();
      }
    });
  }
};

// Register the plugin globally
Chart.register(mutationAnnotationPlugin);

// ---- Linear regression slope helper ----
function _linregSlope(points) {
  // points: [{step, value}, ...] — returns slope (value per step)
  var n = points.length;
  if (n < 2) return 0;
  var sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (var i = 0; i < n; i++) {
    var x = points[i].step;
    var y = points[i].value;
    sumX += x; sumY += y; sumXY += x * y; sumXX += x * x;
  }
  var denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-15) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

// ---- Impact summary computation ----
function computeImpactSummary(rec) {
  var step = rec.step;
  if (step === undefined || step === null) return null;
  var windowSize = 10;
  var results = [];

  S.metricNames.forEach(function(name) {
    var pts = S.metricsData[name] || [];
    if (pts.length < 3) return;

    var before = [];
    var after = [];
    var beforePts = [];
    var afterPts = [];
    pts.forEach(function(p) {
      if (p.step < step && p.step >= step - windowSize) {
        before.push(p.value);
        beforePts.push(p);
      }
      if (p.step > step && p.step <= step + windowSize) {
        after.push(p.value);
        afterPts.push(p);
      }
    });

    if (before.length === 0 || after.length === 0) return;

    var avgBefore = before.reduce(function(a, b) { return a + b; }, 0) / before.length;
    var avgAfter = after.reduce(function(a, b) { return a + b; }, 0) / after.length;
    var change = avgAfter - avgBefore;
    var pctChange = avgBefore !== 0 ? (change / Math.abs(avgBefore)) * 100 : 0;

    // Compute trend (slope) before and after using linear regression
    var slopeBefore = _linregSlope(beforePts);
    var slopeAfter = _linregSlope(afterPts);
    var trendDiff = slopeAfter - slopeBefore;

    // For metrics containing "loss" or "error", decrease is improvement
    var isLossLike = /loss|error|diverge/i.test(name);
    var improving = isLossLike ? change < 0 : change > 0;
    // Trend improvement: for loss-like, a more negative slope after is good
    var trendImproving = isLossLike ? trendDiff < 0 : trendDiff > 0;

    results.push({
      metric: name,
      avgBefore: avgBefore,
      avgAfter: avgAfter,
      change: change,
      pctChange: pctChange,
      improving: improving,
      beforeCount: before.length,
      afterCount: after.length,
      slopeBefore: slopeBefore,
      slopeAfter: slopeAfter,
      trendDiff: trendDiff,
      trendImproving: trendImproving,
    });
  });

  return results;
}

function _dismissImpactPanel() {
  var panel = document.getElementById('impactSummary');
  if (panel) panel.remove();
  _highlightedMutationStep = null;
  if (S.chartInstance) S.chartInstance.update('none');
}

function _fmtSlope(slope) {
  // Format slope as percentage per step
  var pct = (slope * 100);
  var sign = pct >= 0 ? '+' : '';
  if (Math.abs(pct) < 0.01) return '~0/step';
  if (Math.abs(pct) >= 10) return sign + pct.toFixed(0) + '%/step';
  return sign + pct.toFixed(2) + '%/step';
}

function renderImpactSummary(rec) {
  var existing = document.getElementById('impactSummary');
  if (existing) existing.remove();

  var results = computeImpactSummary(rec);
  if (!results || results.length === 0) return;

  var panel = document.createElement('div');
  panel.id = 'impactSummary';
  panel.className = 'impact-summary';

  var header = '<div class="impact-header" id="impactHeader">' +
    '<span>Impact Analysis \u2014 step ' + rec.step + '</span>' +
    '<button class="btn btn-sm impact-close" id="btnCloseImpact">&times;</button>' +
    '</div>';

  var rows = '';
  results.forEach(function(r) {
    var arrow = r.improving ? '\u2191' : '\u2193';
    var arrowClass = r.improving ? 'improving' : 'degrading';
    var pct = Math.abs(r.pctChange).toFixed(1);
    // Trend column
    var trendClass = Math.abs(r.trendDiff) < 1e-8 ? 'neutral' : (r.trendImproving ? 'improving' : 'degrading');
    var trendText = _fmtSlope(r.slopeBefore) + ' \u2192 ' + _fmtSlope(r.slopeAfter);
    rows +=
      '<div class="impact-row">' +
        '<span class="impact-metric">' + r.metric + '</span>' +
        '<span class="impact-before">' + fmtNum(r.avgBefore) + '</span>' +
        '<span class="impact-arrow ' + arrowClass + '">' + arrow + '</span>' +
        '<span class="impact-after">' + fmtNum(r.avgAfter) + '</span>' +
        '<span class="impact-pct ' + arrowClass + '">' + pct + '%</span>' +
        '<span class="impact-trend ' + trendClass + '">' + trendText + '</span>' +
      '</div>';
  });

  panel.innerHTML = header +
    '<div class="impact-columns">' +
      '<span></span><span>Before</span><span></span><span>After</span><span>Change</span><span>Trend</span>' +
    '</div>' + rows;

  var bottomPanel = document.querySelector('.bottom-panel');
  bottomPanel.appendChild(panel);

  // Close on x button click
  document.getElementById('btnCloseImpact').addEventListener('click', function(e) {
    e.stopPropagation();
    _dismissImpactPanel();
  });

  // Close on header click (anywhere on the header row)
  document.getElementById('impactHeader').addEventListener('click', function() {
    _dismissImpactPanel();
  });
}

// ---- Scroll chart to center on a specific step ----
function scrollChartToStep(step) {
  if (!S.chartInstance) return;
  var xScale = S.chartInstance.scales.x;
  if (!xScale) return;
  var range = xScale.max - xScale.min;
  if (range <= 0) return;
  // Only adjust if the step is outside the visible range
  if (step >= xScale.min && step <= xScale.max) return;
  var half = range / 2;
  S.chartInstance.options.scales.x.min = step - half;
  S.chartInstance.options.scales.x.max = step + half;
  S.chartInstance.update('none');
}

function createMetricsChart() {
  var ctx = $('#metricsChart').getContext('2d');
  S.chartInstance = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { type: 'linear', title: {display:true, text:'Step', color:'#7a8fa3', font:{size:11}},
             ticks: {color:'#7a8fa3', font:{size:10}}, grid: {color:'rgba(30,46,68,0.5)'} },
        y: { title: {display:false}, ticks: {color:'#7a8fa3', font:{size:10}},
             grid: {color:'rgba(30,46,68,0.3)'} }
      },
      plugins: {
        legend: {display:false},
        tooltip: { backgroundColor:'#121c2b', borderColor:'#2a4060', borderWidth:1,
                   titleFont:{family:'JetBrains Mono',size:11}, bodyFont:{family:'JetBrains Mono',size:11} }
      },
      elements: { point: {radius:0, hoverRadius:3}, line: {borderWidth:1.5, tension:0.3} }
    }
  });
}

function updateChart() {
  if (!S.chartInstance) return;
  var datasets = [];
  var enabled = new Set();
  S.metricNames.forEach(function(name) {
    if (_metricToggleState[name]) enabled.add(name);
  });

  // In focus mode, only show the focused metric on the main chart
  if (S.focusMetric) {
    enabled = new Set([S.focusMetric]);
  }

  S.metricNames.forEach(function(name) {
    if (!enabled.has(name)) return;
    var pts = S.metricsData[name] || [];
    var color = getColor(name);

    // Live data line
    datasets.push({
      label: name,
      data: pts.map(function(p) { return {x: p.step, y: p.value}; }),
      borderColor: color,
      backgroundColor: 'transparent',
      tension: 0.3,
    });

    var lastStep = pts.length ? pts[pts.length - 1].step : 0;
    var lastVal = pts.length ? pts[pts.length - 1].value : null;
    var cache = _forecastCache[name];

    // Forecast overlay (dotted extension in same color but lighter)
    if (cache && cache.forecast && cache.forecast.values && cache.forecast.values.length) {
      var fc = cache.forecast;
      var fcPts = [{x: lastStep, y: lastVal}];  // Connect to last actual point
      fc.values.forEach(function(v, i) { fcPts.push({x: fc.steps[i], y: v}); });
      datasets.push({
        label: name + ' forecast',
        data: fcPts,
        borderColor: color,
        borderDash: [6, 3],
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 1.2,
        pointRadius: 0,
      });
      // Confidence band
      if (fc.lower && fc.upper) {
        var loPts = [{x: lastStep, y: lastVal}];
        var hiPts = [{x: lastStep, y: lastVal}];
        fc.lower.forEach(function(v, i) { loPts.push({x: fc.steps[i], y: v}); });
        fc.upper.forEach(function(v, i) { hiPts.push({x: fc.steps[i], y: v}); });
        datasets.push({
          label: '_' + name + '_lo',
          data: loPts,
          borderColor: 'transparent',
          backgroundColor: hexToRgba(color, 0.06),
          fill: '+1',
          pointRadius: 0, tension: 0.3,
        });
        datasets.push({
          label: '_' + name + '_hi',
          data: hiPts,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0, tension: 0.3,
        });
      }
    }

    // Mutation impact overlay (cyan dotted)
    if (cache && cache.mutation && cache.mutation.values && cache.mutation.values.length) {
      var mu = cache.mutation;
      var muPts = [{x: mu.fromStep, y: mu.fromVal}];
      mu.values.forEach(function(v, i) { muPts.push({x: mu.steps[i], y: v}); });
      datasets.push({
        label: name + ' post-change',
        data: muPts,
        borderColor: 'rgba(51,204,221,0.7)',
        borderDash: [3, 4],
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 1.2,
        pointRadius: 0,
      });
    }
  });

  S.chartInstance.data.datasets = datasets;
  S.chartInstance.update('none');

  // Also update any pinned metric cards
  updateMetricCards();
}

function hexToRgba(hex, alpha) {
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

var _metricToggleState = {};  // metric name -> boolean (checked)
var _metricDropdownShowAll = false;
var _commonMetricPatterns = [
  'train_loss', 'val_loss', 'loss', 'accuracy', 'val_accuracy',
  'train_accuracy', 'lr', 'learning_rate', 'grad_norm', 'grad_norm_total',
  'epoch', 'perplexity', 'val_perplexity', 'f1', 'val_f1',
  'precision', 'recall', 'auc', 'val_auc', 'bleu'
];

function _isCommonMetric(name) {
  var lower = name.toLowerCase();
  for (var i = 0; i < _commonMetricPatterns.length; i++) {
    if (lower === _commonMetricPatterns[i] || lower.indexOf(_commonMetricPatterns[i]) !== -1) return true;
  }
  return false;
}

function _getVisibleMetrics() {
  var all = [];
  S.metricNames.forEach(function(name) { all.push(name); });
  if (_metricDropdownShowAll || all.length <= 20) return all;
  // Filter to common patterns only
  var filtered = all.filter(_isCommonMetric);
  return filtered.length > 0 ? filtered : all.slice(0, 20);
}

function updateMetricBadge() {
  var container = $('#metricToggles');
  if (!container) return;
  var badge = container.querySelector('.metric-count-badge');
  if (!badge) return;
  var enabledCount = 0;
  var totalCount = 0;
  S.metricNames.forEach(function(name) {
    totalCount++;
    if (_metricToggleState[name]) enabledCount++;
  });
  badge.textContent = enabledCount + '/' + totalCount;
}

var _lastMetricCount = 0;

function updateMetricToggles() {
  var container = $('#metricToggles');
  // Initialize toggle state for new metrics
  var currentCount = 0;
  S.metricNames.forEach(function(name) {
    currentCount++;
    if (!(name in _metricToggleState)) {
      _metricToggleState[name] = true;  // default checked
    }
  });

  // If no new metric names were added, just update the badge
  if (currentCount === _lastMetricCount) {
    updateMetricBadge();
    return;
  }

  // If the dropdown panel is currently open, don't rebuild — just update badge
  var panel = container.querySelector('.metric-dropdown-panel');
  if (panel && panel.classList.contains('open')) {
    updateMetricBadge();
    _lastMetricCount = currentCount;
    return;
  }

  _lastMetricCount = currentCount;
  _renderMetricDropdown(container);
}

function _renderMetricDropdown(container) {
  var totalCount = 0;
  S.metricNames.forEach(function() { totalCount++; });

  // Check if the dropdown was open before re-render
  var wrapper = container.querySelector('.metric-dropdown-wrap');
  var wasOpen = false;
  if (wrapper) {
    var oldPanel = wrapper.querySelector('.metric-dropdown-panel');
    wasOpen = oldPanel && oldPanel.classList.contains('open');
  }

  // Build the dropdown wrapper if it doesn't exist
  if (!wrapper) {
    container.innerHTML = '';
    wrapper = document.createElement('div');
    wrapper.className = 'metric-dropdown-wrap';
    container.appendChild(wrapper);
  }

  // Build toggle button
  var enabledCount = 0;
  S.metricNames.forEach(function(name) {
    if (_metricToggleState[name]) enabledCount++;
  });

  wrapper.innerHTML = '';

  var toggleBtn = document.createElement('button');
  toggleBtn.className = 'btn btn-sm metric-dropdown-btn';
  toggleBtn.innerHTML = 'Metrics <span class="metric-count-badge">' + enabledCount + '/' + totalCount + '</span> &#9662;';
  toggleBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    var panel = wrapper.querySelector('.metric-dropdown-panel');
    if (panel) {
      panel.classList.toggle('open');
    }
  });
  wrapper.appendChild(toggleBtn);

  // Build the dropdown panel
  var panel = document.createElement('div');
  panel.className = 'metric-dropdown-panel' + (wasOpen ? ' open' : '');

  // Controls row: Select All / None / Show All toggle
  var controls = document.createElement('div');
  controls.className = 'metric-dropdown-controls';

  var btnAll = document.createElement('button');
  btnAll.className = 'btn btn-sm';
  btnAll.textContent = 'All On';
  btnAll.addEventListener('click', function(e) {
    e.stopPropagation();
    S.metricNames.forEach(function(name) { _metricToggleState[name] = true; });
    _renderMetricDropdown(container);
    updateChart();
  });

  var btnNone = document.createElement('button');
  btnNone.className = 'btn btn-sm';
  btnNone.textContent = 'All Off';
  btnNone.addEventListener('click', function(e) {
    e.stopPropagation();
    S.metricNames.forEach(function(name) { _metricToggleState[name] = false; });
    _renderMetricDropdown(container);
    updateChart();
  });

  controls.appendChild(btnAll);
  controls.appendChild(btnNone);

  // Show "Show All" toggle only when > 20 metrics
  if (totalCount > 20) {
    var btnShowAll = document.createElement('button');
    btnShowAll.className = 'btn btn-sm' + (_metricDropdownShowAll ? ' btn-accent' : '');
    btnShowAll.textContent = _metricDropdownShowAll ? 'Common Only' : 'Show All';
    btnShowAll.addEventListener('click', function(e) {
      e.stopPropagation();
      _metricDropdownShowAll = !_metricDropdownShowAll;
      _renderMetricDropdown(container);
    });
    controls.appendChild(btnShowAll);
  }

  panel.appendChild(controls);

  // Metric list
  var list = document.createElement('div');
  list.className = 'metric-dropdown-list';

  var visible = _getVisibleMetrics();
  visible.forEach(function(name) {
    var color = getColor(name);
    var row = document.createElement('label');
    row.className = 'metric-dropdown-item';

    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!_metricToggleState[name];
    cb.dataset.metric = name;
    cb.addEventListener('change', function(e) {
      e.stopPropagation();
      _metricToggleState[name] = cb.checked;
      // Update the badge count
      var cnt = 0;
      S.metricNames.forEach(function(n) { if (_metricToggleState[n]) cnt++; });
      var badge = wrapper.querySelector('.metric-count-badge');
      if (badge) badge.textContent = cnt + '/' + totalCount;
      updateChart();
    });

    var swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = color;

    var label = document.createElement('span');
    label.className = 'metric-dropdown-name';
    label.textContent = name;

    var pinBtn = document.createElement('button');
    var isPinned = S.pinnedMetrics && S.pinnedMetrics.has(name);
    pinBtn.className = 'pin-btn' + (isPinned ? ' pin-btn-active' : '');
    pinBtn.dataset.metric = name;
    pinBtn.title = isPinned ? 'Unpin metric card' : 'Pin metric card';
    pinBtn.innerHTML = '&#128204;';
    pinBtn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      toggleMetricCard(name);
      _renderMetricDropdown(container);
    });

    row.appendChild(cb);
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(pinBtn);
    list.appendChild(row);
  });

  if (totalCount > 20 && !_metricDropdownShowAll) {
    var hiddenCount = totalCount - visible.length;
    if (hiddenCount > 0) {
      var note = document.createElement('div');
      note.className = 'metric-dropdown-note';
      note.textContent = '+ ' + hiddenCount + ' more (click "Show All")';
      list.appendChild(note);
    }
  }

  panel.appendChild(list);
  wrapper.appendChild(panel);

  // Close dropdown when clicking outside
  document.addEventListener('click', function closeDropdown(e) {
    if (!wrapper.contains(e.target)) {
      var p = wrapper.querySelector('.metric-dropdown-panel');
      if (p) p.classList.remove('open');
    }
  });
}

// ---- Per-metric pinnable cards ----

function toggleMetricCard(name) {
  if (!S.pinnedMetrics) S.pinnedMetrics = new Set();
  var container = $('#metricCards');
  if (S.pinnedMetrics.has(name)) {
    S.pinnedMetrics.delete(name);
    var existing = container.querySelector('[data-metric-card="' + name + '"]');
    if (existing) existing.remove();
  } else {
    S.pinnedMetrics.add(name);
    createMetricCard(name);
  }
}

function createMetricCard(name) {
  var container = $('#metricCards');
  if (!container) return;
  var card = document.createElement('div');
  card.className = 'card metric-card';
  card.dataset.metricCard = name;
  var color = getColor(name);
  card.innerHTML =
    '<div class="card-header">' +
      '<span style="display:flex;align-items:center;gap:6px">' +
        '<span class="swatch" style="background:' + color + ';width:8px;height:8px;border-radius:2px;display:inline-block"></span>' +
        name +
      '</span>' +
      '<div class="btn-group">' +
        '<button class="btn btn-sm metric-card-focus" title="Focus/expand">&#x26F6;</button>' +
        '<button class="btn btn-sm metric-card-close" title="Unpin">&times;</button>' +
      '</div>' +
    '</div>' +
    '<div class="card-body" style="padding:4px">' +
      '<div style="height:120px;position:relative"><canvas></canvas></div>' +
    '</div>';

  card.querySelector('.metric-card-close').addEventListener('click', function() {
    toggleMetricCard(name);
  });
  card.querySelector('.metric-card-focus').addEventListener('click', function() {
    focusMetricCard(name);
  });

  container.appendChild(card);

  // Create mini chart
  var ctx = card.querySelector('canvas').getContext('2d');
  var miniChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { type: 'linear', display: false },
        y: { ticks: {color:'#7a8fa3', font:{size:9}}, grid: {color:'rgba(30,46,68,0.3)'} }
      },
      plugins: { legend: {display:false} },
      elements: { point: {radius:0}, line: {borderWidth:1.5} }
    }
  });
  if (!S.metricCardCharts) S.metricCardCharts = {};
  S.metricCardCharts[name] = miniChart;
  updateSingleMetricCard(name);
}

function updateSingleMetricCard(name) {
  if (!S.metricCardCharts || !S.metricCardCharts[name]) return;
  var chart = S.metricCardCharts[name];
  var datasets = [];
  var color = getColor(name);
  var pts = S.metricsData[name] || [];

  datasets.push({
    label: name,
    data: pts.map(function(p) { return {x: p.step, y: p.value}; }),
    borderColor: color,
    backgroundColor: 'transparent',
    tension: 0.3,
  });

  var lastStep = pts.length ? pts[pts.length - 1].step : 0;
  var lastVal = pts.length ? pts[pts.length - 1].value : null;
  var cache = _forecastCache[name];

  // Forecast
  if (cache && cache.forecast && cache.forecast.values && cache.forecast.values.length) {
    var fc = cache.forecast;
    var fcPts = [{x: lastStep, y: lastVal}];
    fc.values.forEach(function(v, i) { fcPts.push({x: fc.steps[i], y: v}); });
    datasets.push({
      label: 'forecast',
      data: fcPts,
      borderColor: color, borderDash: [6, 3],
      backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.2, pointRadius: 0,
    });
  }

  // Mutation
  if (cache && cache.mutation && cache.mutation.values && cache.mutation.values.length) {
    var mu = cache.mutation;
    var muPts = [{x: mu.fromStep, y: mu.fromVal}];
    mu.values.forEach(function(v, i) { muPts.push({x: mu.steps[i], y: v}); });
    datasets.push({
      label: 'post-change',
      data: muPts,
      borderColor: 'rgba(51,204,221,0.7)', borderDash: [3, 4],
      backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.2, pointRadius: 0,
    });
  }

  chart.data.datasets = datasets;
  chart.update('none');
}

function updateMetricCards() {
  if (!S.pinnedMetrics) return;
  S.pinnedMetrics.forEach(function(name) {
    updateSingleMetricCard(name);
  });
}

function focusMetricCard(name) {
  // Toggle focus mode on the main chart for this single metric
  if (S.focusMetric === name) {
    S.focusMetric = null;
    document.body.classList.remove('metric-focus-mode');
  } else {
    S.focusMetric = name;
    document.body.classList.add('metric-focus-mode');
  }
  updateChart();
}

// ---- Forecast fetching ----

async function fetchAllForecasts() {
  var promises = [];
  S.metricNames.forEach(function(name) {
    promises.push(fetchForecastForMetric(name));
  });
  await Promise.all(promises);
  updateChart();
}

async function fetchForecastForMetric(name) {
  var data = await api('GET', '/api/projections/forecast/' + encodeURIComponent(name) + '?horizon=30');
  if (!data) return;
  if (!_forecastCache[name]) _forecastCache[name] = {};
  _forecastCache[name].forecast = data;
}

function startForecastPolling() {
  if (_forecastTimer) clearInterval(_forecastTimer);
  _forecastTimer = setInterval(fetchAllForecasts, 10000);
  fetchAllForecasts();
}

// Keep old function name for compatibility
async function updateForecast() {
  await fetchAllForecasts();
}

// Record mutation impact forecast
function recordMutationForecast(step) {
  S.metricNames.forEach(function(name) {
    var pts = S.metricsData[name] || [];
    if (pts.length === 0) return;
    var cache = _forecastCache[name];
    if (cache && cache.forecast) {
      if (!cache.mutation) cache.mutation = {};
      cache.mutation = {
        fromStep: step,
        fromVal: pts[pts.length - 1].value,
        steps: cache.forecast.steps ? cache.forecast.steps.slice() : [],
        values: cache.forecast.values ? cache.forecast.values.slice() : [],
      };
    }
  });
}
