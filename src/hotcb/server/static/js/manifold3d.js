/**
 * hotcb dashboard — Three.js 3D manifold/feature rendering
 */

function init3DScene(containerId) {
  var container = document.getElementById(containerId);
  if (!container || !window.THREE) {
    if (container) container.innerHTML = '<div class="manifold-fallback">3D not available</div>';
    return null;
  }
  var w = container.clientWidth || 300;
  var h = container.clientHeight || 200;

  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a1018);

  var camera = new THREE.PerspectiveCamera(50, w/h, 0.1, 1000);
  camera.position.set(3, 2, 4);
  camera.lookAt(0, 0, 0);

  var renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  scene.add(new THREE.GridHelper(6, 12, 0x1e2e44, 0x121c2b));
  scene.add(new THREE.AxesHelper(2));

  // Mouse rotation state
  var isDragging = false, prevX = 0, prevY = 0;
  var rotX = 0.3, rotY = 0.5;

  renderer.domElement.addEventListener('mousedown', function(e) {
    isDragging = true; prevX = e.clientX; prevY = e.clientY;
  });
  window.addEventListener('mouseup', function() { isDragging = false; });
  renderer.domElement.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    rotY += (e.clientX - prevX) * 0.008;
    rotX += (e.clientY - prevY) * 0.008;
    rotX = Math.max(-1.2, Math.min(1.2, rotX));
    prevX = e.clientX; prevY = e.clientY;
  });
  renderer.domElement.addEventListener('wheel', function(e) {
    camera.position.multiplyScalar(e.deltaY > 0 ? 1.05 : 0.95);
    e.preventDefault();
  }, {passive: false});

  var ctx = {
    scene: scene, camera: camera, renderer: renderer, animId: null,
    getRotX: function() { return rotX; },
    getRotY: function() { return rotY; },
  };

  // Start render loop
  (function loop() {
    var r = 5;
    camera.position.x = r * Math.sin(rotY) * Math.cos(rotX);
    camera.position.y = r * Math.sin(rotX);
    camera.position.z = r * Math.cos(rotY) * Math.cos(rotX);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    ctx.animId = requestAnimationFrame(loop);
  })();

  return ctx;
}

function render3DPoints(ctx, points, interventionSteps) {
  if (!ctx) return;
  var scene = ctx.scene;

  // Remove old data points
  var old = scene.children.filter(function(c) { return c.userData && c.userData.isDataPoint; });
  old.forEach(function(o) { scene.remove(o); });

  if (!points || points.length === 0) return;

  var intSet = new Set(interventionSteps || []);
  var positions = [];
  var colors = [];

  for (var i = 0; i < points.length; i++) {
    var p = points[i];
    positions.push(p[0] || 0, p[1] || 0, p[2] || 0);
    var t = i / Math.max(points.length - 1, 1);
    if (intSet.has(i)) {
      colors.push(1, 0.3, 0.37);
    } else {
      // Multi-stop rainbow gradient: blue → cyan → green → yellow → orange → red
      var r, g, b;
      if (t < 0.2) {
        // blue → cyan
        var s = t / 0.2;
        r = 0; g = s; b = 1;
      } else if (t < 0.4) {
        // cyan → green
        var s = (t - 0.2) / 0.2;
        r = 0; g = 1; b = 1 - s;
      } else if (t < 0.6) {
        // green → yellow
        var s = (t - 0.4) / 0.2;
        r = s; g = 1; b = 0;
      } else if (t < 0.8) {
        // yellow → orange
        var s = (t - 0.6) / 0.2;
        r = 1; g = 1 - s * 0.35; b = 0;
      } else {
        // orange → red
        var s = (t - 0.8) / 0.2;
        r = 1; g = 0.65 - s * 0.65; b = 0;
      }
      colors.push(r, g, b);
    }
  }

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  var pts = new THREE.Points(geo, new THREE.PointsMaterial({size:3, vertexColors:true, sizeAttenuation:true}));
  pts.userData.isDataPoint = true;
  scene.add(pts);

  // Trajectory line
  var lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  var line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({color:0x00d4aa, opacity:0.3, transparent:true}));
  line.userData.isDataPoint = true;
  scene.add(line);
}

async function fetchManifold() {
  var method = $('#manifoldMethod').value;
  var data = await api('GET', '/api/manifolds/metric?method=' + method + '&n_components=3');
  if (!data) return;

  if (!S.manifoldCtx) {
    S.manifoldCtx = init3DScene('manifold-3d');
    if (!S.manifoldCtx) return;
  }

  var points = data.points || data.embedding || [];
  var interventions = data.intervention_indices || [];
  render3DPoints(S.manifoldCtx, points, interventions);

  // Update manifold info panel
  var metricNames = data.metric_names || [];
  var originalDims = metricNames.length;
  var activeSubtab = document.querySelector('[data-subtab].active');
  var isFeatureSpace = activeSubtab && activeSubtab.getAttribute('data-subtab') === 'feature-space';
  var infoMethod = $('#manifoldInfoMethod');
  var infoSpace = $('#manifoldInfoSpace');
  var infoPoints = $('#manifoldInfoPoints');
  var infoDims = $('#manifoldInfoDims');
  if (infoMethod) infoMethod.textContent = (data.method || method).toUpperCase();
  if (infoSpace) infoSpace.textContent = isFeatureSpace ? 'Feature Space' : 'Metric Space';
  if (infoPoints) infoPoints.textContent = points.length;
  if (infoDims) infoDims.textContent = originalDims + 'D \u2192 3D';
}

async function fetchFeatures() {
  var data = await api('GET', '/api/features/snapshots?last_n=100');
  var container = document.getElementById('feature-3d');
  if (!data || !data.snapshots || data.snapshots.length === 0) {
    if (container) container.innerHTML = '<div class="manifold-fallback">No feature captures. Enable with kernel.enable_feature_capture()</div>';
    return;
  }

  if (!S.featureCtx) {
    S.featureCtx = init3DScene('feature-3d');
    if (!S.featureCtx) return;
  }

  var points = [];
  var featureDims = 0;
  data.snapshots.forEach(function(snap) {
    var acts = snap.activations || [];
    if (acts.length === 0) return;
    if (acts[0] && acts[0].length > featureDims) featureDims = acts[0].length;
    var avg = [0, 0, 0];
    acts.forEach(function(row) {
      avg[0] += (row[0] || 0) / acts.length;
      avg[1] += (row[1] || 0) / acts.length;
      avg[2] += (row[2] || 0) / acts.length;
    });
    points.push(avg);
  });
  render3DPoints(S.featureCtx, points, []);

  // Update manifold info panel for feature space
  var infoMethod = $('#manifoldInfoMethod');
  var infoSpace = $('#manifoldInfoSpace');
  var infoPoints = $('#manifoldInfoPoints');
  var infoDims = $('#manifoldInfoDims');
  if (infoMethod) infoMethod.textContent = 'AVG';
  if (infoSpace) infoSpace.textContent = 'Feature Space';
  if (infoPoints) infoPoints.textContent = points.length;
  if (infoDims) infoDims.textContent = featureDims + 'D \u2192 3D';
}
