/**
 * hotcb dashboard — WebSocket connection and message handling
 */

function connectWS() {
  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var ws = new WebSocket(proto + '//' + location.host + '/ws');
  S.ws = ws;

  ws.onopen = function() {
    $('#wsStatus').className = 'status-dot ok';
    $('#wsLabel').textContent = 'connected';
    ws.send(JSON.stringify({channels: ['metrics', 'applied', 'mutations', 'segments']}));
  };

  ws.onclose = function() {
    $('#wsStatus').className = 'status-dot error';
    $('#wsLabel').textContent = 'disconnected';
    setTimeout(connectWS, 3000);
  };

  ws.onerror = function() {
    $('#wsStatus').className = 'status-dot error';
    $('#wsLabel').textContent = 'error';
  };

  ws.onmessage = function(e) {
    try {
      var msg = JSON.parse(e.data);
      var ch = msg.channel;
      var data = msg.data;
      if (!Array.isArray(data)) return;

      if (ch === 'metrics') {
        var prevSize = S.metricNames.size;
        data.forEach(function(rec) {
          var step = rec.step || 0;
          var metrics = rec.metrics || {};
          Object.assign(S.latestMetrics, metrics);
          Object.keys(metrics).forEach(function(name) {
            var value = metrics[name];
            if (typeof value !== 'number') return;
            S.metricNames.add(name);
            if (!S.metricsData[name]) S.metricsData[name] = [];
            S.metricsData[name].push({step: step, value: value});
            if (S.metricsData[name].length > 5000) {
              S.metricsData[name] = S.metricsData[name].slice(-4000);
            }
          });
        });
        if (S.metricNames.size !== prevSize) {
          updateMetricToggles();
        } else {
          updateMetricBadge();
        }
        updateChart();
        computeHealth();
        // Update step counter
        var maxStep = 0;
        S.metricNames.forEach(function(name) {
          var pts = S.metricsData[name] || [];
          if (pts.length) maxStep = Math.max(maxStep, pts[pts.length-1].step);
        });
        var stepEl = document.getElementById('stepValue');
        if (stepEl) stepEl.textContent = maxStep;
        // Trigger forecast refresh on new data
        if (typeof onNewMetricsForForecast === 'function') onNewMetricsForForecast(maxStep);
      }

      if (ch === 'applied') {
        // On initial WebSocket burst, clear existing timeline to avoid duplicates
        // from the REST initialLoad() that may have already populated it
        if (msg.initial) {
          S.appliedData = [];
          clearTimelineDedup();
          var tl = $('#timelineList');
          if (tl) tl.innerHTML = '';
          $('#mutationCount').textContent = '0';
        }
        data.forEach(function(rec) {
          S.appliedData.push(rec);
          addTimelineItem(rec);
          // Sync slider knobs from applied params
          if (rec.params) {
            syncSlidersFromApplied(rec.params);
          }
        });
      }
    } catch(err) { console.error('WS parse error:', err); }
  };
}
