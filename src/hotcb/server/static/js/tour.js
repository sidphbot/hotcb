/**
 * hotcb dashboard — First-time walkthrough tour (progressive disclosure)
 */

var TOUR_STEPS = [
  {
    target: '.logo',
    title: 'Welcome to hotcb',
    text: 'This is your live training control plane. Watch metrics flow in real-time, adjust hyperparameters on the fly, and see instant forecasts of how changes affect training.',
    position: 'bottom'
  },
  {
    target: '#wsStatus',
    title: 'Connection Status',
    text: 'This indicator shows whether the dashboard is connected to your training process via WebSocket. Green = live, yellow = reconnecting.',
    position: 'bottom'
  },
  {
    target: '#stepCounter',
    title: 'Training Step',
    text: 'Current training step. Updates in real-time as your model trains.',
    position: 'bottom'
  },
  {
    target: '#metricsChart',
    title: 'Live Metrics',
    text: 'All your training metrics plotted live. Toggle individual metrics with the checkboxes above. Forecasts appear as dotted extensions.',
    position: 'right'
  },
  {
    target: '#metricToggles',
    title: 'Metric Controls',
    text: 'Toggle metrics on/off. Click the pin icon next to a metric name to create a dedicated card for it with its own forecast overlay.',
    position: 'bottom'
  },
  {
    target: '#knobPanel',
    title: 'Hyperparameter Knobs',
    text: 'Adjust learning rate, weight decay, and loss weights. Changes are applied instantly when you click "Apply".',
    position: 'left'
  },
  {
    target: '#btnApply',
    title: 'Apply Changes',
    text: 'Send your current knob values to the training process. A cyan dotted line will show the pre-change forecast for comparison.',
    position: 'left'
  },
  {
    target: '#btnSchedule',
    title: 'Schedule Changes',
    text: 'Plan changes for a future step. Use this to set up learning rate schedules or planned interventions.',
    position: 'left'
  },
  {
    target: '#timelineList',
    title: 'Mutation Timeline',
    text: 'Every change you make is recorded here with its step number. Click on a mutation to see its impact on the forecast.',
    position: 'top'
  },
  {
    target: '#autopilotMode',
    title: 'Autopilot',
    text: '"Suggest" mode will propose interventions when it detects issues (plateau, divergence, overfitting). "Auto" mode applies high-confidence fixes automatically.',
    position: 'left'
  },
  {
    target: '#themeSelect',
    title: 'Themes',
    text: 'Switch between Midnight (dark), Light, Solarized, and Paper themes.',
    position: 'bottom'
  }
];

var _tourStep = 0;
var _tourActive = false;
var _tourOverlay = null;
var _tourHighlight = null;
var _tourTooltip = null;
var _tourCurrentTarget = null;

function shouldShowTour() {
  return !localStorage.getItem('hotcb-tour-seen');
}

function startTour() {
  _tourStep = 0;
  _tourActive = true;

  // Create overlay elements
  _tourOverlay = document.createElement('div');
  _tourOverlay.className = 'tour-overlay';
  _tourOverlay.addEventListener('click', function(e) {
    if (e.target === _tourOverlay) nextTourStep();
  });
  document.body.appendChild(_tourOverlay);

  _tourHighlight = document.createElement('div');
  _tourHighlight.className = 'tour-highlight';
  document.body.appendChild(_tourHighlight);

  _tourTooltip = document.createElement('div');
  _tourTooltip.className = 'tour-tooltip';
  document.body.appendChild(_tourTooltip);

  showTourStep(0);
}

function showTourStep(idx) {
  if (idx >= TOUR_STEPS.length) {
    endTour();
    return;
  }
  // Remove highlight class from previous target
  if (_tourCurrentTarget) {
    _tourCurrentTarget.classList.remove('tour-target-highlight');
    _tourCurrentTarget = null;
  }

  _tourStep = idx;
  var step = TOUR_STEPS[idx];
  var target = document.querySelector(step.target);

  if (!target) {
    // Skip missing elements
    showTourStep(idx + 1);
    return;
  }

  // Lift the target element above the overlay so it appears sharp
  target.classList.add('tour-target-highlight');
  _tourCurrentTarget = target;

  var rect = target.getBoundingClientRect();
  var pad = 6;

  // Position highlight
  _tourHighlight.style.left = (rect.left - pad) + 'px';
  _tourHighlight.style.top = (rect.top - pad) + 'px';
  _tourHighlight.style.width = (rect.width + pad * 2) + 'px';
  _tourHighlight.style.height = (rect.height + pad * 2) + 'px';

  // Build tooltip content
  _tourTooltip.innerHTML =
    '<h4>' + step.title + '</h4>' +
    '<p>' + step.text + '</p>' +
    '<div class="tour-actions">' +
      '<button class="tour-skip" onclick="endTour()">Skip tour</button>' +
      '<div style="display:flex;gap:8px;align-items:center">' +
        '<span class="tour-progress">' + (idx + 1) + '/' + TOUR_STEPS.length + '</span>' +
        (idx > 0 ? '<button class="btn btn-sm" onclick="prevTourStep()">Back</button>' : '') +
        '<button class="btn btn-sm btn-accent" onclick="nextTourStep()">' +
          (idx === TOUR_STEPS.length - 1 ? 'Finish' : 'Next') +
        '</button>' +
      '</div>' +
    '</div>';

  // Position tooltip
  var tw = 320;
  var th = _tourTooltip.offsetHeight || 150;
  var left, top;

  switch (step.position) {
    case 'bottom':
      left = rect.left + rect.width / 2 - tw / 2;
      top = rect.bottom + pad + 12;
      break;
    case 'top':
      left = rect.left + rect.width / 2 - tw / 2;
      top = rect.top - pad - th - 12;
      break;
    case 'left':
      left = rect.left - tw - pad - 12;
      top = rect.top + rect.height / 2 - th / 2;
      break;
    case 'right':
      left = rect.right + pad + 12;
      top = rect.top + rect.height / 2 - th / 2;
      break;
    default:
      left = rect.left;
      top = rect.bottom + 12;
  }

  // Keep on screen
  left = Math.max(10, Math.min(left, window.innerWidth - tw - 20));
  top = Math.max(10, Math.min(top, window.innerHeight - th - 20));

  _tourTooltip.style.left = left + 'px';
  _tourTooltip.style.top = top + 'px';
}

function nextTourStep() {
  showTourStep(_tourStep + 1);
}

function prevTourStep() {
  if (_tourStep > 0) showTourStep(_tourStep - 1);
}

function endTour() {
  _tourActive = false;
  localStorage.setItem('hotcb-tour-seen', '1');
  if (_tourCurrentTarget) { _tourCurrentTarget.classList.remove('tour-target-highlight'); _tourCurrentTarget = null; }
  if (_tourOverlay) { _tourOverlay.remove(); _tourOverlay = null; }
  if (_tourHighlight) { _tourHighlight.remove(); _tourHighlight = null; }
  if (_tourTooltip) { _tourTooltip.remove(); _tourTooltip = null; }
}

// Make functions globally available
window.nextTourStep = nextTourStep;
window.prevTourStep = prevTourStep;
window.endTour = endTour;
window.startTour = startTour;
