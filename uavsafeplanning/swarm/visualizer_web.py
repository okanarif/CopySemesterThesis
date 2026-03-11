"""Generate a standalone HTML + Three.js 3D simulation for multi-UAV trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .uav import UAV

_COLORS_HEX = [
    "0x1f77b4", "0xff7f0e", "0x2ca02c", "0xd62728", "0x9467bd",
    "0x8c564b", "0xe377c2", "0x7f7f7f", "0xbcbd22", "0x17becf",
]


def _serialize_uav(uav: UAV, idx: int, dt: float = 0.05) -> dict:
    """Convert a planned UAV into a JSON-serialisable dict of sampled frames."""
    t_max = uav.time_offset + uav.total_time()
    times = np.arange(0.0, t_max + dt, dt)
    positions = [uav.sample_position(t).tolist() for t in times]
    return {
        "id": uav.uav_id,
        "color": _COLORS_HEX[idx % len(_COLORS_HEX)],
        "radius": float(uav.radius),
        "offset": float(uav.time_offset),
        "start": uav.start.tolist(),
        "goal": uav.goal.tolist(),
        "times": times.tolist(),
        "positions": positions,
    }


def _serialize_obstacles(obs_np: np.ndarray) -> list:
    """Down-sample the obstacle cloud to keep the HTML manageable."""
    if len(obs_np) > 4000:
        idx = np.random.default_rng(42).choice(len(obs_np), 4000, replace=False)
        obs_np = obs_np[idx]
    return obs_np.tolist()


def generate_simulation_html(
    uavs: List[UAV],
    output_path: str = "simulation.html",
    obs_np: Optional[np.ndarray] = None,
    dt: float = 0.05,
    safety_margin: float = 0.0,
) -> str:
    """Write a self-contained HTML file with an animated Three.js scene.

    Parameters
    ----------
    uavs : list of UAV
        Planned UAVs with populated ``sto_results``.
    output_path : str
        File path for the generated HTML.
    obs_np : ndarray, optional
        Obstacle points (N, 3) – rendered as small cubes.
    dt : float
        Time resolution for frame sampling.
    safety_margin : float
        Shown in the HUD.

    Returns
    -------
    str : absolute path to the written file.
    """
    uav_data = [_serialize_uav(u, i, dt=dt) for i, u in enumerate(uavs)]
    t_global_max = max(u.time_offset + u.total_time() for u in uavs)
    obs_data = _serialize_obstacles(obs_np) if obs_np is not None else []

    html = _HTML_TEMPLATE.replace("__UAV_DATA__", json.dumps(uav_data))
    html = html.replace("__OBS_DATA__", json.dumps(obs_data))
    html = html.replace("__T_MAX__", f"{t_global_max:.4f}")
    html = html.replace("__DT__", f"{dt:.4f}")
    html = html.replace("__SAFETY__", f"{safety_margin:.2f}")

    out = Path(output_path).resolve()
    out.write_text(html, encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------------------
# HTML template (Three.js r160 via CDN)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Multi-UAV Swarm Simulation</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { overflow:hidden; background:#1a1a2e; font-family: 'Segoe UI', sans-serif; }
  canvas { display:block; }
  #hud {
    position:absolute; top:12px; left:12px; color:#eee; font-size:13px;
    background:rgba(0,0,0,0.55); padding:10px 14px; border-radius:8px;
    pointer-events:none; line-height:1.6;
  }
  #controls {
    position:absolute; bottom:16px; left:50%; transform:translateX(-50%);
    display:flex; align-items:center; gap:12px;
    background:rgba(0,0,0,0.6); padding:8px 18px; border-radius:10px;
  }
  #controls button {
    background:#444; color:#fff; border:none; padding:6px 14px;
    border-radius:5px; cursor:pointer; font-size:13px;
  }
  #controls button:hover { background:#666; }
  #slider { width:400px; accent-color:#17becf; }
  #timeLabel { color:#ccc; font-size:13px; min-width:80px; text-align:center; }
  #speedLabel { color:#999; font-size:12px; }
</style>
</head>
<body>
<div id="hud"></div>
<div id="controls">
  <button id="btnPlay">Play</button>
  <button id="btnStep">&gt;</button>
  <input id="slider" type="range" min="0" max="1000" value="0"/>
  <span id="timeLabel">0.00 s</span>
  <button id="btnSlower">-</button>
  <span id="speedLabel">1.0x</span>
  <button id="btnFaster">+</button>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const UAV_DATA = __UAV_DATA__;
const OBS_DATA = __OBS_DATA__;
const T_MAX    = __T_MAX__;
const DT       = __DT__;
const SAFETY   = __SAFETY__;

// --- Scene setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 200, 500);

const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 1000);
camera.position.set(50, 80, 120);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(50, 50, 10);
controls.enableDamping = true;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(60, 100, 80);
scene.add(dirLight);

// Ground plane
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(200, 200),
  new THREE.MeshStandardMaterial({ color:0x2a2a40, roughness:0.9 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.set(50, -0.5, 50);
scene.add(ground);

// Grid helper
const grid = new THREE.GridHelper(200, 40, 0x444466, 0x333355);
grid.position.set(50, -0.4, 50);
scene.add(grid);

// Obstacles
const obsMat = new THREE.MeshStandardMaterial({ color:0x3366aa, transparent:true, opacity:0.25 });
const obsGeo = new THREE.BoxGeometry(1, 1, 1);
const obsMesh = new THREE.InstancedMesh(obsGeo, obsMat, OBS_DATA.length);
OBS_DATA.forEach((p, i) => {
  const m = new THREE.Matrix4().setPosition(p[0], p[2], p[1]);
  obsMesh.setMatrixAt(i, m);
});
obsMesh.instanceMatrix.needsUpdate = true;
scene.add(obsMesh);

// --- UAV objects ---
const uavMeshes = [];
const trailLines = [];
const warningRings = [];

UAV_DATA.forEach((u, i) => {
  const col = parseInt(u.color);

  // Sphere body
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(u.radius * 0.6, 20, 20),
    new THREE.MeshStandardMaterial({ color: col, metalness:0.3, roughness:0.5 })
  );
  scene.add(mesh);
  uavMeshes.push(mesh);

  // Trail
  const trailGeo = new THREE.BufferGeometry();
  const maxPts = u.positions.length;
  const trailPos = new Float32Array(maxPts * 3);
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setDrawRange(0, 0);
  const trailLine = new THREE.Line(
    trailGeo,
    new THREE.LineBasicMaterial({ color: col, transparent:true, opacity:0.6 })
  );
  scene.add(trailLine);
  trailLines.push(trailLine);

  // Collision warning ring
  const ringGeo = new THREE.RingGeometry(u.radius * 1.2, u.radius * 1.6, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color:0xff0000, transparent:true, opacity:0, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);
  warningRings.push(ring);

  // Start / goal markers
  const sph = (pos, c) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.6, 12, 12), new THREE.MeshStandardMaterial({ color:c }));
    m.position.set(pos[0], pos[2], pos[1]);
    scene.add(m);
  };
  sph(u.start, 0x00ff88);
  sph(u.goal, 0xff4444);
});

// --- Interpolation helper ---
function getPos(uavIdx, t) {
  const u = UAV_DATA[uavIdx];
  const times = u.times;
  const positions = u.positions;
  if (t <= times[0]) return positions[0];
  if (t >= times[times.length-1]) return positions[positions.length-1];
  let lo = 0, hi = times.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid;
  }
  const frac = (t - times[lo]) / (times[hi] - times[lo]);
  const a = positions[lo], b = positions[hi];
  return [
    a[0] + (b[0]-a[0]) * frac,
    a[1] + (b[1]-a[1]) * frac,
    a[2] + (b[2]-a[2]) * frac,
  ];
}

function frameIndex(uavIdx, t) {
  const times = UAV_DATA[uavIdx].times;
  let lo = 0;
  for (let i = 0; i < times.length; i++) {
    if (times[i] <= t) lo = i; else break;
  }
  return lo;
}

// --- Animation state ---
let playing = false;
let speed = 1.0;
let currentTime = 0;

const btnPlay  = document.getElementById('btnPlay');
const btnStep  = document.getElementById('btnStep');
const slider   = document.getElementById('slider');
const timeLabel = document.getElementById('timeLabel');
const speedLabel = document.getElementById('speedLabel');
const hud      = document.getElementById('hud');

btnPlay.onclick  = () => { playing = !playing; btnPlay.textContent = playing ? 'Pause' : 'Play'; };
btnStep.onclick  = () => { playing = false; currentTime = Math.min(currentTime + DT * 5, T_MAX); };
document.getElementById('btnSlower').onclick = () => { speed = Math.max(0.1, speed - 0.25); speedLabel.textContent = speed.toFixed(2) + 'x'; };
document.getElementById('btnFaster').onclick = () => { speed = Math.min(10, speed + 0.25); speedLabel.textContent = speed.toFixed(2) + 'x'; };
slider.oninput = () => { currentTime = (slider.value / 1000) * T_MAX; };

let lastRAF = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dtReal = (now - lastRAF) / 1000;
  lastRAF = now;

  if (playing) {
    currentTime += dtReal * speed;
    if (currentTime > T_MAX) { currentTime = 0; }
  }

  slider.value = (currentTime / T_MAX) * 1000;
  timeLabel.textContent = currentTime.toFixed(2) + ' s';

  // Update UAV positions and trails
  let hudText = `<b>Swarm Simulation</b><br>UAVs: ${UAV_DATA.length} &nbsp; Safety: ${SAFETY} m<br><br>`;

  let minPairDist = Infinity;

  UAV_DATA.forEach((u, i) => {
    const p = getPos(i, currentTime);
    uavMeshes[i].position.set(p[0], p[2], p[1]);
    warningRings[i].position.set(p[0], p[2] + 0.05, p[1]);

    // Update trail
    const fi = frameIndex(i, currentTime);
    const trailAttr = trailLines[i].geometry.attributes.position;
    for (let k = 0; k <= fi && k < u.positions.length; k++) {
      const pp = u.positions[k];
      trailAttr.array[k*3]   = pp[0];
      trailAttr.array[k*3+1] = pp[2];
      trailAttr.array[k*3+2] = pp[1];
    }
    trailAttr.needsUpdate = true;
    trailLines[i].geometry.setDrawRange(0, fi + 1);

    hudText += `<span style="color:${('#' + u.color.slice(2))}">\u25CF</span> ${u.id}: offset=${u.offset.toFixed(1)}s<br>`;
  });

  // Collision warnings
  for (let i = 0; i < UAV_DATA.length; i++) {
    warningRings[i].material.opacity = 0;
    for (let j = i+1; j < UAV_DATA.length; j++) {
      const pi = getPos(i, currentTime);
      const pj = getPos(j, currentTime);
      const d = Math.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2);
      minPairDist = Math.min(minPairDist, d);
      const threshold = UAV_DATA[i].radius + UAV_DATA[j].radius + SAFETY;
      if (d < threshold * 1.5) {
        const intensity = Math.max(0, 1 - d / (threshold * 1.5));
        warningRings[i].material.opacity = Math.max(warningRings[i].material.opacity, intensity * 0.8);
        warningRings[j].material.opacity = Math.max(warningRings[j].material.opacity, intensity * 0.8);
      }
    }
  }

  if (UAV_DATA.length >= 2) {
    hudText += `<br>Min pair dist: ${minPairDist.toFixed(2)} m`;
  }
  hud.innerHTML = hudText;

  controls.update();
  renderer.render(scene, camera);
}

requestAnimationFrame(animate);
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""
