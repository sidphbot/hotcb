/**
 * hotcb dashboard — WebSocket connection and message handling
 */

var _wsRetryCount = 0;
var _wsMaxRetries = 20;
var _slidersInitialized = false;
var _lastSeenMaxStep = 0;  // track for run-reset detection

function connectWS() {
  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var ws = new WebSocket(proto + '//' + location.host + '/ws');
  S.ws = ws;

  ws.onopen = function() {
    _wsRetryCount = 0;
    $('#wsStatus').className = 'status-dot ok';
    $('#wsLabel').textContent = 'connected';
    ws.send(JSON.stringify({channels: ['metrics', 'applied', 'mutations', 'segments']}));
  };

  ws.onclose = function() {
    $('#wsStatus').className = 'status-dot error';
    $('#wsLabel').textContent = 'disconnected';
    if (_wsRetryCount < _wsMaxRetries) {
      var delay = Math.min(3000 * Math.pow(1.5, _wsRetryCount), 30000);
      _wsRetryCount++;
      setTimeout(connectWS, delay);
    }
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
        if (typeof dismissChartWaiting === 'function') dismissChartWaiting();
        // Detect run reset: if incoming steps jump backwards, clear stale state
        if (data.length > 0) {
          var firstIncoming = data[0].step || 0;
          if (_lastSeenMaxStep > 0 && firstIncoming < _lastSeenMaxStep - 10) {
            // Steps went backwards — new run started. Flush stale caches.
            S.metricsData = {};
            S.appliedData = [];
            S.metricNames = new Set();
            S.latestMetrics = {};
            if (typeof _forecastCache !== 'undefined') _forecastCache = {};
            if (typeof _highlightedMutationStep !== 'undefined') _highlightedMutationStep = null;
            _lastSeenMaxStep = 0;
            _slidersInitialized = false;
            var tl = document.getElementById('timelineList');
            if (tl) tl.innerHTML = '';
            if (typeof clearTimelineDedup === 'function') clearTimelineDedup();
            var mc = document.getElementById('mutationCount');
            if (mc) mc.textContent = '0';
          }
        }
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
            // No client-side cap — keep all points.
            // Downsampling for rendering happens in updateChart().
          });
        });
        if (S.metricNames.size !== prevSize) {
          updateMetricToggles();
        } else {
          updateMetricBadge();
        }
        updateChart();
        computeHealth();
        // Sync slider knobs from initial metrics (once per training session)
        if (!_slidersInitialized && data.length > 0) {
          _slidersInitialized = true;
          var last = data[data.length - 1];
          var lm = last.metrics || {};
          var syncObj = {};
          if (lm.lr && lm.lr > 0) syncObj.lr = lm.lr;
          if (lm.weight_decay && lm.weight_decay > 0) syncObj.weight_decay = lm.weight_decay;
          if (Object.keys(syncObj).length > 0) syncSlidersFromApplied(syncObj);
        }
        // Update step counter
        var maxStep = 0;
        S.metricNames.forEach(function(name) {
          var pts = S.metricsData[name] || [];
          if (pts.length) maxStep = Math.max(maxStep, pts[pts.length-1].step);
        });
        _lastSeenMaxStep = maxStep;
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
          // Sync slider knobs from applied params (check both fields)
          var syncParams = rec.params || rec.payload;
          if (syncParams) {
            syncSlidersFromApplied(syncParams);
          }
        });
      }
    } catch(err) { console.error('WS parse error:', err); }
  };
}
