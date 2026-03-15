/**
 * hotcb dashboard — Unified chart system with per-metric cards
 */

// Forecast update interval
var _forecastTimer = null;
var _forecastCache = {};  // metric -> {forecast, mutation}

// Highlighted mutation step (set when user clicks a timeline item)
var _highlightedMutationStep = null;

// Step range control: 'all' | 'last200' | 'last500' | {min, max}
var _chartStepRange = 'all';

// Y-axis normalization: when enabled, each metric is normalized to [0,1]
var _chartNormalize = false;

// Max points to render per dataset (avoids sluggish charts on very long runs)
var _maxRenderPoints = (S.config && S.config.chart) ? S.config.chart.max_render_points : 2000;

/**
 * LTTB (Largest-Triangle-Three-Buckets) downsampling.
 * Takes [{x, y}] and returns a reduced array preserving visual shape.
 */
function _lttbDownsample(data, threshold) {
  if (data.length <= threshold) return data;
  var sampled = [data[0]];  // always keep first
  var bucketSize = (data.length - 2) / (threshold - 2);
  var a = 0;  // index of previously selected point
  for (var i = 0; i < threshold - 2; i++) {
    // Calculate bucket range
    var bStart = Math.floor((i + 1) * bucketSize) + 1;
    var bEnd   = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length - 1);
    // Average of next bucket (for triangle area calc)
    var avgX = 0, avgY = 0, cnt = 0;
    var nbStart = Math.floor((i + 2) * bucketSize) + 1;
    var nbEnd   = Math.min(Math.floor((i + 3) * bucketSize) + 1, data.length - 1);
    if (nbStart > data.length - 1) { nbStart = data.length - 1; nbEnd = data.length - 1; }
    for (var j = nbStart; j <= nbEnd; j++) { avgX += data[j].x; avgY += data[j].y; cnt++; }
    avgX /= cnt; avgY /= cnt;
    // Pick point in current bucket with largest triangle area
    var maxArea = -1, maxIdx = bStart;
    var ax = data[a].x, ay = data[a].y;
    for (var k = bStart; k <= bEnd; k++) {
      var area = Math.abs((ax - avgX) * (data[k].y - ay) - (ax - data[k].x) * (avgY - ay));
      if (area > maxArea) { maxArea = area; maxIdx = k; }
    }
    sampled.push(data[maxIdx]);
    a = maxIdx;
  }
  sampled.push(data[data.length - 1]);  // always keep last
  return sampled;
}

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

    // Determine actual data range — skip annotations outside metric data
    var dataMinStep = _getMinStep();
    var dataMaxStep = _getMaxStep();
    if (dataMaxStep === 0) return;  // no metric data yet

    // Collect visible annotations with pixel positions for staggering
    var annotations = [];
    S.appliedData.forEach(function(rec) {
      var step = rec.step;
      if (step === undefined || step === null) return;
      // Filter to both visible x-axis range AND actual data range
      if (step < xScale.min || step > xScale.max) return;
      if (step < dataMinStep || step > dataMaxStep) return;
      var x = xScale.getPixelForValue(step);
      annotations.push({rec: rec, x: x, step: step});
    });

    // Sort by x position for overlap detection
    annotations.sort(function(a, b) { return a.x - b.x; });

    // Assign stagger rows: if labels overlap horizontally, push down
    var rowHeight = 15;
    var maxRows = 4;
    for (var i = 0; i < annotations.length; i++) {
      annotations[i].row = 0;
      // Check overlap with previously placed labels
      for (var r = 0; r < maxRows; r++) {
        var overlaps = false;
        for (var j = 0; j < i; j++) {
          if (annotations[j].row === r && Math.abs(annotations[i].x - annotations[j].x) < 70) {
            overlaps = true;
            break;
          }
        }
        if (!overlaps) { annotations[i].row = r; break; }
        annotations[i].row = r + 1;
      }
    }

    annotations.forEach(function(ann) {
      var rec = ann.rec;
      var x = ann.x;
      var isHighlighted = (_highlightedMutationStep === ann.step);

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

      // Build compact label — split into multiple lines if needed
      var lines = [];
      var annotParams = (rec.params && typeof rec.params === 'object') ? rec.params :
                        (rec.payload && typeof rec.payload === 'object') ? rec.payload : null;
      if (annotParams) {
        var keys = Object.keys(annotParams);
        keys.slice(0, 3).forEach(function(k) {
          var v = annotParams[k];
          if (typeof v === 'number') {
            v = v < 0.01 || v > 1e4 ? v.toExponential(1) : parseFloat(v.toPrecision(3));
          } else if (typeof v === 'object' && v !== null) {
            v = JSON.stringify(v);
          }
          lines.push(k + '\u2192' + v);
        });
      } else if (rec.op) {
        lines.push(rec.op);
      }

      if (lines.length > 0) {
        ctx.save();
        ctx.font = (isHighlighted ? 'bold ' : '') + '9px "JetBrains Mono", monospace';
        var pad = 3;
        var lineH = 11;
        var baseY = top - 2 - ann.row * rowHeight;

        // Draw each line stacked upward
        for (var li = lines.length - 1; li >= 0; li--) {
          var text = lines[li];
          var textWidth = ctx.measureText(text).width;
          var labelX = x - textWidth / 2 - pad;
          var labelY = baseY - (lines.length - 1 - li) * lineH;

          // Background pill
          ctx.fillStyle = isHighlighted
            ? 'rgba(255, 152, 51, 0.25)'
            : 'rgba(255, 152, 51, 0.12)';
          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(labelX, labelY - lineH, textWidth + pad * 2, lineH + 2, 3);
          } else {
            ctx.rect(labelX, labelY - lineH, textWidth + pad * 2, lineH + 2);
          }
          ctx.fill();

          // Text
          ctx.fillStyle = isHighlighted
            ? 'rgba(255, 200, 120, 1)'
            : 'rgba(255, 200, 120, 0.7)';
          ctx.textBaseline = 'bottom';
          ctx.fillText(text, labelX + pad, labelY);
        }
        ctx.restore();
      }
    });
  }
};

// Register the plugin globally
Chart.register(mutationAnnotationPlugin);

// Custom tooltip positioner — show to the right of the cursor
Chart.Tooltip.positioners.rightOfCursor = function(elements, eventPosition) {
  return {
    x: eventPosition.x + 15,
    y: eventPosition.y
  };
};

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

    // Use direction inference to determine if change is improvement
    var dir = (typeof inferMetricDirection === 'function') ? inferMetricDirection(name) : (/loss|error|diverge/i.test(name) ? 'min' : 'max');
    var isMinimize = (dir === 'min');
    var improving = isMinimize ? change < 0 : change > 0;
    // Trend improvement: for minimize, a more negative slope after is good
    var trendImproving = isMinimize ? trendDiff < 0 : trendDiff > 0;

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
  if (range <= 0) range = 200;
  // Only adjust if the step is outside the visible range
  if (step >= xScale.min && step <= xScale.max) return;
  var half = range / 2;
  _chartStepRange = { min: Math.max(0, step - half), max: step + half };
  _applyChartStepRange();
  S.chartInstance.update('none');
  _updateRangeButtons();
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
        tooltip: { backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--bg-card').trim() || '#121c2b',
                   borderColor: getComputedStyle(document.documentElement).getPropertyValue('--border-bright').trim() || '#2a4060', borderWidth:1,
                   titleFont:{family:'JetBrains Mono',size:11}, bodyFont:{family:'JetBrains Mono',size:11},
                   usePointStyle: true, boxWidth: 8, boxHeight: 8,
                   intersect: false, mode: 'nearest', axis: 'x', position: 'rightOfCursor',
                   itemSort: function(a, b, data) {
                     // Sort tooltip items by Y-pixel distance to cursor (TensorBoard-style)
                     var chart = a.chart || (data && data.chart);
                     var cursorY = 0;
                     if (chart && chart._lastEvent) cursorY = chart._lastEvent.y;
                     var ay = a.element ? a.element.y : 0;
                     var by = b.element ? b.element.y : 0;
                     return Math.abs(ay - cursorY) - Math.abs(by - cursorY);
                   },
                   callbacks: {
                     label: function(ctx) {
                       var ds = ctx.dataset;
                       var label = ds.label || '';
                       // Skip internal datasets (confidence bands)
                       if (label.startsWith('_')) return null;
                       var rawY = ctx.parsed.y;
                       // When normalized, show both normalized and raw values
                       if (_chartNormalize) {
                         var metricName = label.replace(/ forecast$/, '').replace(/ post-change$/, '');
                         var rawPts = S.metricsData[metricName];
                         if (rawPts && rawPts.length > 0) {
                           // Find raw value at nearest step
                           var step = ctx.parsed.x;
                           var rawVal = null;
                           for (var i = 0; i < rawPts.length; i++) {
                             if (rawPts[i].step >= step) { rawVal = rawPts[i].value; break; }
                             rawVal = rawPts[i].value;
                           }
                           if (rawVal !== null) {
                             return label + ': ' + fmtNum(rawVal) + ' (norm: ' + rawY.toFixed(3) + ')';
                           }
                         }
                       }
                       return label + ': ' + fmtNum(rawY);
                     }
                   }
        }
      },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      elements: { point: {radius:0, hoverRadius:5, hitRadius:10}, line: {borderWidth:1.5, tension: (S.config && S.config.chart) ? S.config.chart.line_tension : 0.15} }
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

    // Compute per-metric min/max for normalization
    var metricMin = Infinity, metricMax = -Infinity;
    if (_chartNormalize && pts.length > 0) {
      pts.forEach(function(p) {
        if (p.value < metricMin) metricMin = p.value;
        if (p.value > metricMax) metricMax = p.value;
      });
      if (metricMax === metricMin) { metricMin -= 0.5; metricMax += 0.5; }
    }
    var _normFn = (_chartNormalize && metricMax !== metricMin)
      ? function(v) { return (v - metricMin) / (metricMax - metricMin); }
      : function(v) { return v; };

    // Live data line (LTTB-downsampled for rendering performance)
    var chartPts = pts.map(function(p) { return {x: p.step, y: _normFn(p.value)}; });
    chartPts = _lttbDownsample(chartPts, _maxRenderPoints);
    var _cfgTension = (S.config && S.config.chart) ? S.config.chart.line_tension : 0.15;
    var _cfgForecastDash = (S.config && S.config.chart) ? S.config.chart.forecast_dash : [6, 3];
    var _cfgMutationDash = (S.config && S.config.chart) ? S.config.chart.mutation_dash : [3, 4];

    datasets.push({
      label: name,
      data: chartPts,
      borderColor: color,
      backgroundColor: 'transparent',
      tension: _cfgTension,
    });

    var lastStep = pts.length ? pts[pts.length - 1].step : 0;
    var lastVal = pts.length ? pts[pts.length - 1].value : null;
    var lastValNorm = lastVal !== null ? _normFn(lastVal) : null;
    var cache = _forecastCache[name];
    var showOverlays = S.focusMetric === name || (S.pinnedMetrics && S.pinnedMetrics.has(name));

    // Forecast overlay — only shown for pinned or focused metrics to reduce clutter
    // Guard: only render if forecast steps are contiguous with current data (avoids stale cross-run connectors)
    if (showOverlays && cache && cache.forecast && cache.forecast.values && cache.forecast.values.length
        && lastStep > 0 && lastVal !== null
        && cache.forecast.steps && cache.forecast.steps[0] <= lastStep + 50) {
      var fc = cache.forecast;
      var fcPts = [{x: lastStep, y: lastValNorm}];  // Connect to last actual point
      fc.values.forEach(function(v, i) { fcPts.push({x: fc.steps[i], y: _normFn(v)}); });
      datasets.push({
        label: name + ' forecast',
        data: fcPts,
        borderColor: color,
        borderDash: _cfgForecastDash,
        backgroundColor: 'transparent',
        tension: _cfgTension,
        borderWidth: 1.2,
        pointRadius: 0,
      });
      // Confidence band
      if (fc.lower && fc.upper) {
        var loPts = [{x: lastStep, y: lastValNorm}];
        var hiPts = [{x: lastStep, y: lastValNorm}];
        fc.lower.forEach(function(v, i) { loPts.push({x: fc.steps[i], y: _normFn(v)}); });
        fc.upper.forEach(function(v, i) { hiPts.push({x: fc.steps[i], y: _normFn(v)}); });
        datasets.push({
          label: '_' + name + '_lo',
          data: loPts,
          borderColor: 'transparent',
          backgroundColor: hexToRgba(color, 0.06),
          fill: '+1',
          pointRadius: 0, tension: _cfgTension,
        });
        datasets.push({
          label: '_' + name + '_hi',
          data: hiPts,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0, tension: _cfgTension,
        });
      }
    }

    // Mutation impact overlay — only shown for pinned or focused metrics
    // Guard: only render if fromStep is within current data range (avoids stale cross-run connectors)
    if (showOverlays && cache && cache.mutation && cache.mutation.values && cache.mutation.values.length) {
      var mu = cache.mutation;
      var inDataRange = pts.length > 0 && mu.fromStep >= pts[0].step && mu.fromStep <= pts[pts.length - 1].step;
      if (inDataRange) {
        var muPts = [{x: mu.fromStep, y: _normFn(mu.fromVal)}];
        mu.values.forEach(function(v, i) { muPts.push({x: mu.steps[i], y: _normFn(v)}); });
        datasets.push({
          label: name + ' post-change',
          data: muPts,
          borderColor: 'rgba(51,204,221,0.7)',
          borderDash: _cfgMutationDash,
          backgroundColor: 'transparent',
          tension: _cfgTension,
          borderWidth: 1.2,
          pointRadius: 0,
        });
      }
    }
  });

  S.chartInstance.data.datasets = datasets;

  // Apply step range to x-axis
  _applyChartStepRange();

  // Update Y-axis title for normalization mode
  var yOpts = S.chartInstance.options.scales.y;
  if (_chartNormalize) {
    yOpts.title = {display: true, text: 'Normalized [0,1]', color: '#7a8fa3', font: {size: 10}};
  } else {
    yOpts.title = {display: false};
  }

  S.chartInstance.update('none');

  // Also update any pinned metric cards
  updateMetricCards();
}

function _getMaxStep() {
  var maxStep = 0;
  S.metricNames.forEach(function(name) {
    var pts = S.metricsData[name] || [];
    if (pts.length) maxStep = Math.max(maxStep, pts[pts.length - 1].step);
  });
  return maxStep;
}

function _getMinStep() {
  var minStep = Infinity;
  S.metricNames.forEach(function(name) {
    var pts = S.metricsData[name] || [];
    if (pts.length) minStep = Math.min(minStep, pts[0].step);
  });
  return minStep === Infinity ? 0 : minStep;
}

function _applyChartStepRange() {
  if (!S.chartInstance) return;
  var xOpts = S.chartInstance.options.scales.x;
  if (_chartStepRange === 'all') {
    delete xOpts.min;
    delete xOpts.max;
  } else if (_chartStepRange === 'last200') {
    var mx = _getMaxStep();
    xOpts.min = Math.max(0, mx - 200);
    delete xOpts.max;
  } else if (_chartStepRange === 'last500') {
    var mx2 = _getMaxStep();
    xOpts.min = Math.max(0, mx2 - 500);
    delete xOpts.max;
  } else if (typeof _chartStepRange === 'object' && _chartStepRange !== null) {
    xOpts.min = _chartStepRange.min;
    xOpts.max = _chartStepRange.max;
  }
}

function setChartStepRange(mode) {
  _chartStepRange = mode;
  updateChart();
  _updateRangeButtons();
}

function _updateRangeButtons() {
  var btns = document.querySelectorAll('#stepRangeControls .range-btn[data-range]');
  btns.forEach(function(btn) {
    var isActive = (typeof _chartStepRange === 'string' && btn.dataset.range === _chartStepRange);
    btn.classList.toggle('active', isActive);
  });
  // Update custom inputs if range is a custom object
  var minEl = document.getElementById('rangeMin');
  var maxEl = document.getElementById('rangeMax');
  if (minEl && maxEl) {
    if (typeof _chartStepRange === 'object' && _chartStepRange !== null) {
      minEl.value = _chartStepRange.min != null ? _chartStepRange.min : '';
      maxEl.value = _chartStepRange.max != null ? _chartStepRange.max : '';
    }
  }
}

function initStepRangeControls() {
  var container = document.getElementById('stepRangeControls');
  if (!container) return;

  // Preset buttons
  container.querySelectorAll('.range-btn[data-range]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      setChartStepRange(btn.dataset.range);
    });
  });

  // Custom range apply
  var applyBtn = document.getElementById('rangeApply');
  var minEl = document.getElementById('rangeMin');
  var maxEl = document.getElementById('rangeMax');
  if (applyBtn && minEl && maxEl) {
    applyBtn.addEventListener('click', function() {
      var mn = minEl.value !== '' ? parseInt(minEl.value, 10) : undefined;
      var mx = maxEl.value !== '' ? parseInt(maxEl.value, 10) : undefined;
      if (mn === undefined && mx === undefined) {
        setChartStepRange('all');
      } else {
        setChartStepRange({
          min: mn !== undefined ? mn : 0,
          max: mx !== undefined ? mx : undefined
        });
      }
    });
    // Enter key in inputs triggers apply
    [minEl, maxEl].forEach(function(el) {
      el.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') applyBtn.click();
      });
    });
  }
}

function hexToRgba(hex, alpha) {
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

var _metricToggleState = {};  // metric name -> boolean (checked)
var _metricDropdownShowAll = false;
var _dropdownCloseHandler = null;
var _commonMetricPatterns = [
  'train_loss', 'val_loss', 'loss', 'accuracy', 'val_accuracy',
  'train_accuracy', 'lr', 'learning_rate', 'grad_norm', 'grad_norm_total',
  'epoch', 'perplexity', 'val_perplexity', 'f1', 'val_f1',
  'precision', 'recall', 'auc', 'val_auc', 'bleu'
];

// Metrics shown by default on chart — losses and key metric only
var _defaultOnPatterns = ['loss', 'val_loss', 'train_loss'];

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
      // Default: only show losses and key metric; others off
      var lower = name.toLowerCase();
      var isDefault = false;
      for (var di = 0; di < _defaultOnPatterns.length; di++) {
        if (lower === _defaultOnPatterns[di] || lower.indexOf(_defaultOnPatterns[di]) !== -1) { isDefault = true; break; }
      }
      // Also check AI key metric
      var aiKeyMetricEl = document.getElementById('aiKeyMetric');
      if (aiKeyMetricEl && aiKeyMetricEl.value === name) isDefault = true;
      _metricToggleState[name] = isDefault;
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
    var row = document.createElement('div');
    row.className = 'metric-dropdown-item';

    var isActive = !!_metricToggleState[name];
    var isPinned = S.pinnedMetrics && S.pinnedMetrics.has(name);

    // Filled/hollow dot toggle (replaces checkbox + swatch)
    var dot = document.createElement('span');
    dot.className = 'metric-dot' + (isActive ? ' active' : '') + (isPinned ? ' pinned' : '');
    dot.style.color = color;
    dot.style.borderColor = color;
    if (isActive) dot.style.background = color;
    dot.title = isActive ? 'Hide metric' : 'Show metric';
    dot.addEventListener('click', function(e) {
      e.stopPropagation();
      _metricToggleState[name] = !_metricToggleState[name];
      var cnt = 0;
      S.metricNames.forEach(function(n) { if (_metricToggleState[n]) cnt++; });
      var badge = wrapper.querySelector('.metric-count-badge');
      if (badge) badge.textContent = cnt + '/' + totalCount;
      _renderMetricDropdown(container);
      updateChart();
    });
    // Double-click or right-click to toggle pin
    dot.addEventListener('dblclick', function(e) {
      e.stopPropagation(); e.preventDefault();
      toggleMetricCard(name);
      _renderMetricDropdown(container);
    });
    dot.addEventListener('contextmenu', function(e) {
      e.preventDefault(); e.stopPropagation();
      toggleMetricCard(name);
      _renderMetricDropdown(container);
    });

    var label = document.createElement('span');
    label.className = 'metric-dropdown-name';
    label.textContent = name;
    label.addEventListener('click', function(e) {
      e.stopPropagation();
      _metricToggleState[name] = !_metricToggleState[name];
      var cnt = 0;
      S.metricNames.forEach(function(n) { if (_metricToggleState[n]) cnt++; });
      var badge = wrapper.querySelector('.metric-count-badge');
      if (badge) badge.textContent = cnt + '/' + totalCount;
      _renderMetricDropdown(container);
      updateChart();
    });

    row.appendChild(dot);
    row.appendChild(label);

    // Pin indicator — visible icon when metric is pinned
    if (isPinned) {
      var pinIcon = document.createElement('span');
      pinIcon.className = 'metric-pin-icon';
      pinIcon.textContent = '\u{1F4CC}';
      pinIcon.title = 'Pinned \u2014 double-click dot to unpin';
      pinIcon.addEventListener('click', function(e) {
        e.stopPropagation();
        toggleMetricCard(name);
        _renderMetricDropdown(container);
      });
      row.appendChild(pinIcon);
    }

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

  // Close dropdown when clicking outside (single delegated listener)
  if (!_dropdownCloseHandler) {
    _dropdownCloseHandler = function(e) {
      var wraps = document.querySelectorAll('.metric-dropdown-wrap');
      wraps.forEach(function(w) {
        if (!w.contains(e.target)) {
          var p = w.querySelector('.metric-dropdown-panel');
          if (p) p.classList.remove('open');
        }
      });
    };
    document.addEventListener('click', _dropdownCloseHandler);
  }
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
        '<button class="btn btn-sm metric-card-key" title="Set as key metric" style="font-size:9px;color:var(--yellow,#facc15)">&#x2605;</button>' +
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
  card.querySelector('.metric-card-key').addEventListener('click', function() {
    var dir = (typeof inferMetricDirection === 'function') ? inferMetricDirection(name) : 'min';
    var dirLabel = (dir === 'min') ? 'minimize (lower=better)' : 'maximize (higher=better)';
    if (confirm('Set "' + name + '" as the key metric?\nDetected direction: ' + dirLabel + '\n\nYou can change the direction in the AI config panel.')) {
      api('POST', '/api/autopilot/ai/key_metric', {metric: name, mode: 'auto'}).then(function(res) {
        if (res && res.status === 'updated') {
          var keyMetricEl = document.getElementById('aiKeyMetric');
          if (keyMetricEl) keyMetricEl.value = name;
          var modeEl = document.getElementById('aiKeyMetricMode');
          if (modeEl) modeEl.value = res.mode || 'auto';
          // Visual feedback
          var stars = document.querySelectorAll('.metric-card-key');
          stars.forEach(function(s) { s.style.color = 'var(--yellow,#facc15)'; s.style.fontWeight = ''; });
          var thisCard = card.querySelector('.metric-card-key');
          if (thisCard) { thisCard.style.color = 'var(--green,#4ade80)'; thisCard.style.fontWeight = 'bold'; }
        } else {
          alert('Failed to set key metric: ' + ((res && res.error) || 'unknown error'));
        }
      });
    }
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

  var _cfgTension = (S.config && S.config.chart) ? S.config.chart.line_tension : 0.15;
  var _cfgForecastDash = (S.config && S.config.chart) ? S.config.chart.forecast_dash : [6, 3];
  var _cfgMutationDash = (S.config && S.config.chart) ? S.config.chart.mutation_dash : [3, 4];

  var chartPts = pts.map(function(p) { return {x: p.step, y: p.value}; });
  chartPts = _lttbDownsample(chartPts, 500);  // mini cards need fewer points
  datasets.push({
    label: name,
    data: chartPts,
    borderColor: color,
    backgroundColor: 'transparent',
    tension: _cfgTension,
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
      borderColor: color, borderDash: _cfgForecastDash,
      backgroundColor: 'transparent', tension: _cfgTension, borderWidth: 1.2, pointRadius: 0,
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
      borderColor: 'rgba(51,204,221,0.7)', borderDash: _cfgMutationDash,
      backgroundColor: 'transparent', tension: _cfgTension, borderWidth: 1.2, pointRadius: 0,
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
var _forecastInFlight = false;
var _forecastPendingRefresh = false;
var _lastForecastStep = 0;

async function fetchAllForecasts() {
  // Debounce: don't pile up concurrent forecast requests
  if (_forecastInFlight) {
    _forecastPendingRefresh = true;
    return;
  }
  _forecastInFlight = true;
  // Only fetch forecasts for pinned or focused metrics — not all metrics
  var names = [];
  if (S.focusMetric) {
    names.push(S.focusMetric);
  }
  if (S.pinnedMetrics) {
    S.pinnedMetrics.forEach(function(name) {
      if (names.indexOf(name) === -1) names.push(name);
    });
  }
  if (names.length === 0) {
    _forecastInFlight = false;
    return;
  }
  var batchSize = (S.config && S.config.ui) ? S.config.ui.forecast_batch_size : 8;
  for (var i = 0; i < names.length; i += batchSize) {
    var batch = names.slice(i, i + batchSize);
    await Promise.all(batch.map(fetchForecastForMetric));
  }
  _forecastInFlight = false;
  updateChart();
  // If new data arrived while we were fetching, refresh again
  if (_forecastPendingRefresh) {
    _forecastPendingRefresh = false;
    setTimeout(fetchAllForecasts, 1000);
  }
}

async function fetchForecastForMetric(name) {
  var data = await api('GET', '/api/projections/forecast/' + encodeURIComponent(name) + '?horizon=30');
  if (!data) return;
  if (!_forecastCache[name]) _forecastCache[name] = {};
  _forecastCache[name].forecast = data;
}

function startForecastPolling() {
  if (_forecastTimer) clearInterval(_forecastTimer);
  var _forecastPollMs = (S.config && S.config.ui) ? S.config.ui.forecast_poll_interval : 5000;
  // Poll at configured interval — new metrics also trigger refresh via onNewMetricsForForecast
  _forecastTimer = setInterval(function() {
    // Only fetch if we have pinned/focused metrics to avoid wasted work
    if ((S.pinnedMetrics && S.pinnedMetrics.size > 0) || S.focusMetric) {
      fetchAllForecasts();
    }
  }, _forecastPollMs);
  // Delay initial fetch to let the dashboard settle
  setTimeout(function() {
    if ((S.pinnedMetrics && S.pinnedMetrics.size > 0) || S.focusMetric) {
      fetchAllForecasts();
    }
  }, 3000);
}

// Called from websocket when new metrics arrive — trigger forecast refresh
function onNewMetricsForForecast(maxStep) {
  // Only trigger every N steps (from config), and only if there are pinned/focused metrics
  var _forecastCadence = (S.config && S.config.ui) ? S.config.ui.forecast_step_cadence : 10;
  if (maxStep - _lastForecastStep >= _forecastCadence) {
    _lastForecastStep = maxStep;
    if ((S.pinnedMetrics && S.pinnedMetrics.size > 0) || S.focusMetric) {
      fetchAllForecasts();
    }
  }
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
