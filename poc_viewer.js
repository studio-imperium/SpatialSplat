import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const MODEL_WORLD_SCALE = 0.96;
const MODEL_WORLD_TRANSLATION_Y = -0.25;
const POINT_LIMIT = 16000;

const viewer = document.querySelector("#viewer");
const sceneSelect = document.querySelector("#scene-select");
const variantSelect = document.querySelector("#variant-select");
const seedControl = document.querySelector("#seed-control");
const seedSelect = document.querySelector("#seed-select");
const splatMode = document.querySelector("#splat-mode");
const splatModeControl = splatMode.closest("label");
const representationLegend = document.querySelector("#representation-legend");
const metricsTable = document.querySelector("#metrics");
const inputImage = document.querySelector("#input-image");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e1e1e);
const overlayScene = new THREE.Scene();

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewer.clientWidth, viewer.clientHeight, false);
viewer.appendChild(renderer.domElement);

const spark = new SparkRenderer({ renderer });
scene.add(spark);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomToCursor = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(3, 4, 2);
scene.add(light);

const solidGroup = new THREE.Group();
const wireGroup = new THREE.Group();
const splatGroup = new THREE.Group();
scene.add(solidGroup, splatGroup);
overlayScene.add(wireGroup);

const solidMaterial = new THREE.MeshStandardMaterial({
  color: 0x999999,
  roughness: 0.8,
  transparent: true,
  opacity: 0.75,
  polygonOffset: true,
  polygonOffsetFactor: 1,
  polygonOffsetUnits: 1,
});
const wireMaterial = new THREE.LineBasicMaterial({
  color: 0xff0000,
  depthTest: false,
  depthWrite: false,
});
const pointMaterial = new THREE.PointsMaterial({
  color: 0x00bcd4,
  size: 2,
  sizeAttenuation: false,
  transparent: true,
  opacity: 0.65,
  depthWrite: false,
});

let currentCameraData = null;
let actualSplat = null;
let actualSplatRoot = null;
let loadVersion = 0;
let freshSceneNames = new Set();
let baselineSceneNames = new Set();
let pairSceneNames = new Set();
let sceneRoots = new Map();
let freshResultRoots = new Map();

function isFreshVariant() {
  return variantSelect.value === "fresh_base" || variantSelect.value === "fresh_lora";
}

function isInputVariant() {
  return variantSelect.value === "input_pair";
}

function updateResultControls() {
  const sceneName = sceneSelect.value;
  const support = {
    baseline: baselineSceneNames.has(sceneName),
    optimized: baselineSceneNames.has(sceneName),
    input_pair: pairSceneNames.has(sceneName),
    fresh_base: freshSceneNames.has(sceneName),
    fresh_lora: freshSceneNames.has(sceneName),
  };
  for (const option of variantSelect.options) {
    option.disabled = !support[option.value];
  }
  if (variantSelect.selectedOptions[0]?.disabled) {
    variantSelect.value = support.input_pair
      ? "input_pair"
      : [...variantSelect.options].find(option => !option.disabled)?.value || "";
  }
  const fresh = isFreshVariant();
  seedControl.hidden = !fresh;
  updateRepresentation();
}

function prettyName(value) {
  return value
    .replace(/^\d+_/, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, character => character.toUpperCase());
}

function clearGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.geometry?.dispose();
  }
}

function primitiveGeometry(primitive) {
  if (primitive.kind === "box") {
    return new THREE.BoxGeometry(...primitive.size);
  }
  if (primitive.kind === "sphere") {
    return new THREE.SphereGeometry(Math.min(...primitive.size) * 0.5, 24, 16);
  }
  if (primitive.kind === "cylinder") {
    const radius = Math.min(primitive.size[0], primitive.size[2]) * 0.5;
    return new THREE.CylinderGeometry(radius, radius, primitive.size[1], 24, 1);
  }
  throw new Error(`Unsupported primitive: ${primitive.kind}`);
}

function buildPrimitives(primitives) {
  clearGroup(solidGroup);
  clearGroup(wireGroup);

  for (const primitive of primitives) {
    const geometry = primitiveGeometry(primitive);
    const solid = new THREE.Mesh(geometry, solidMaterial);
    solid.position.fromArray(primitive.center);
    solid.rotation.y = THREE.MathUtils.degToRad(primitive.yaw_degrees || 0);
    solidGroup.add(solid);

    const edgeGeometry = primitive.kind === "box"
      ? new THREE.EdgesGeometry(geometry)
      : new THREE.WireframeGeometry(geometry);
    const wire = new THREE.LineSegments(edgeGeometry, wireMaterial);
    wire.position.copy(solid.position);
    wire.rotation.copy(solid.rotation);
    wire.renderOrder = 1000;
    wireGroup.add(wire);
  }
}

function clearActualSplat() {
  if (actualSplatRoot) scene.remove(actualSplatRoot);
  actualSplat?.dispose?.();
  actualSplat = null;
  actualSplatRoot = null;
}

function buildActualSplat(url) {
  clearActualSplat();
  actualSplat = new SplatMesh({ url });
  actualSplat.rotation.y = Math.PI / 2;

  actualSplatRoot = new THREE.Group();
  actualSplatRoot.add(actualSplat);
  actualSplatRoot.rotation.x = Math.PI;
  actualSplatRoot.scale.setScalar(MODEL_WORLD_SCALE);
  actualSplatRoot.position.y = MODEL_WORLD_TRANSLATION_Y;
  scene.add(actualSplatRoot);
  updateRepresentation();
}

function updateRepresentation() {
  const inputPair = isInputVariant();
  const showSplats = splatMode.checked;
  splatModeControl.hidden = inputPair;
  splatGroup.visible = !inputPair && !showSplats;
  solidGroup.visible = inputPair || !showSplats;
  if (actualSplatRoot) actualSplatRoot.visible = !inputPair && showSplats;
  representationLegend.innerHTML = inputPair
    ? "generated image + primitives"
    : showSplats
      ? "actual splat"
      : '<span class="swatch cyan"></span> points';
}

function findPlyDataOffset(bytes) {
  const marker = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114, 10];
  outer: for (let index = 0; index <= bytes.length - marker.length; index++) {
    for (let offset = 0; offset < marker.length; offset++) {
      if (bytes[index + offset] !== marker[offset]) continue outer;
    }
    return index + marker.length;
  }
  throw new Error("Invalid PLY header");
}

async function loadSplat(url, version) {
  buildActualSplat(url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load PLY: ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (version !== loadVersion) return;

  const bytes = new Uint8Array(buffer);
  const dataOffset = findPlyDataOffset(bytes);
  const header = new TextDecoder("ascii").decode(bytes.subarray(0, dataOffset));
  const lines = header.split(/\r?\n/);
  let vertexCount = 0;
  let readingVertices = false;
  const properties = [];

  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      vertexCount = Number(line.split(" ").at(-1));
      readingVertices = true;
    } else if (line.startsWith("element ") && !line.startsWith("element vertex ")) {
      readingVertices = false;
    } else if (readingVertices && line.startsWith("property float ")) {
      properties.push(line.split(" ").at(-1));
    }
  }

  const propertyIndex = Object.fromEntries(
    properties.map((name, index) => [name, index])
  );
  for (const name of ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]) {
    if (!(name in propertyIndex)) throw new Error(`PLY is missing ${name}`);
  }

  const stride = properties.length * 4;
  const data = new DataView(buffer, dataOffset);
  const samples = new Array(vertexCount);

  for (let index = 0; index < vertexCount; index++) {
    const base = index * stride;
    const read = name => data.getFloat32(base + propertyIndex[name] * 4, true);
    const opacity = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, read("opacity")))));
    samples[index] = {
      x: read("x"),
      y: read("y"),
      z: read("z"),
      importance: opacity * Math.exp(read("scale_0") + read("scale_1") + read("scale_2")),
    };
  }

  samples.sort((left, right) => right.importance - left.importance);
  const selected = samples.slice(0, Math.min(samples.length, POINT_LIMIT));
  const positions = new Float32Array(selected.length * 3);

  for (let index = 0; index < selected.length; index++) {
    const sample = selected[index];
    positions[index * 3] = sample.z * MODEL_WORLD_SCALE;
    positions[index * 3 + 1] = -sample.y * MODEL_WORLD_SCALE + MODEL_WORLD_TRANSLATION_Y;
    positions[index * 3 + 2] = sample.x * MODEL_WORLD_SCALE;
  }

  clearGroup(splatGroup);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  splatGroup.add(new THREE.Points(geometry, pointMaterial));
}

function updateProjection() {
  const width = Math.max(viewer.clientWidth, 1);
  const height = Math.max(viewer.clientHeight, 1);
  const aspect = width / height;
  const baseSize = currentCameraData?.ortho_scale || 1.45;
  const verticalSize = baseSize * Math.max(1, 1 / aspect);
  camera.left = -verticalSize * aspect * 0.5;
  camera.right = verticalSize * aspect * 0.5;
  camera.top = verticalSize * 0.5;
  camera.bottom = -verticalSize * 0.5;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function resetCamera() {
  const cameraData = currentCameraData || {
    position: [1.7, 1.45, 1.7],
    target: [0, -0.08, 0],
    ortho_scale: 1.45,
  };
  camera.position.fromArray(cameraData.position);
  camera.zoom = 1;
  controls.target.fromArray(cameraData.target);
  updateProjection();
  controls.update();
}

function updateMetrics(metrics) {
  metricsTable.hidden = !metrics;
  if (!metrics) return;
  document.querySelector("#spatial-score").textContent = metrics.spatial_score.toFixed(4);
  document.querySelector("#depth-loss").textContent = metrics.depth_loss.toFixed(4);
  document.querySelector("#mask-iou").textContent = metrics.soft_iou.toFixed(4);
  document.querySelector("#median-depth").textContent = metrics.median_normalized_depth_error.toFixed(4);
  document.querySelector("#p95-depth").textContent = metrics.p95_normalized_depth_error.toFixed(4);
}

async function loadScene(sceneName) {
  const version = ++loadVersion;
  status.hidden = false;
  errorBox.hidden = true;
  try {
    const root = sceneRoots.get(sceneName) || `poc_data/${sceneName}`;
    const sceneResponse = await fetch(`${root}/scene.json`);
    if (!sceneResponse.ok) throw new Error(`Failed to load scene: ${sceneResponse.status}`);
    const sceneData = await sceneResponse.json();
    if (version !== loadVersion) return;

    currentCameraData = sceneData.camera;
    buildPrimitives(sceneData.primitives);

    if (isInputVariant()) {
      clearActualSplat();
      clearGroup(splatGroup);
      updateMetrics(null);
      inputImage.src = `${root}/generated_image.png`;
      inputImage.hidden = false;
      resetCamera();
      updateRepresentation();
      status.hidden = true;
      return;
    }

    inputImage.hidden = true;
    inputImage.removeAttribute("src");
    const optimizationResponse = await fetch(`${root}/latent_optimization_summary.json`);
    const optimizedOption = variantSelect.querySelector('option[value="optimized"]');
    optimizedOption.disabled = !optimizationResponse.ok;
    if (!optimizationResponse.ok && variantSelect.value === "optimized") {
      variantSelect.value = "baseline";
    }
    const optimization = optimizationResponse.ok
      ? await optimizationResponse.json()
      : null;
    let metrics;
    let splatUrl;

    if (isFreshVariant()) {
      if (!freshSceneNames.has(sceneName)) {
        throw new Error("No fresh-generation test exists for this scene");
      }
      const seed = seedSelect.value.padStart(4, "0");
      const freshRoot = `${freshResultRoots.get(sceneName)}/${sceneName}/seed_${seed}`;
      const freshMetrics = await fetch(`${freshRoot}/metrics.json`).then(response => {
        if (!response.ok) throw new Error(`Failed to load fresh metrics: ${response.status}`);
        return response.json();
      });
      const resultName = variantSelect.value === "fresh_lora" ? "lora" : "base";
      metrics = freshMetrics[resultName].aggregate;
      splatUrl = `${freshRoot}/${resultName}_splat.ply`;
    } else {
      const optimized = variantSelect.value === "optimized" && optimization;
      metrics = optimized
        ? optimization.fresh_anchors.optimized
        : await fetch(`${root}/base_metrics.json`).then(response => response.json());
      splatUrl = `${root}/${optimized ? "optimized_splat.ply" : "base_splat.ply"}`;
    }

    updateMetrics(metrics);
    resetCamera();
    await loadSplat(splatUrl, version);
    if (version === loadVersion) status.hidden = true;
  } catch (error) {
    console.error(error);
    status.hidden = true;
    errorBox.textContent = error.message || "Failed to load scene";
    errorBox.hidden = false;
  }
}

async function initialize() {
  try {
    const [
      baselineResponse,
      freshResponse,
      heldoutResponse,
      diverseTestResponse,
      diverseSplitResponse,
    ] = await Promise.all([
      fetch("poc_data/baseline_summary.json"),
      fetch("poc_data/fresh_generation_six_view/summary.json"),
      fetch("poc_data/heldout_alpha_generation_six_view/summary.json"),
      fetch("poc_data/diverse_test_final/summary.json"),
      fetch("poc_data/diverse_train/split.json"),
    ]);
    if (!baselineResponse.ok) throw new Error("Failed to load baseline summary");
    const summary = await baselineResponse.json();
    baselineSceneNames = new Set(summary.scenes.map(result => result.scene));
    for (const sceneName of baselineSceneNames) pairSceneNames.add(sceneName);
    if (freshResponse.ok) {
      const freshSummary = await freshResponse.json();
      for (const pair of freshSummary.pairs) {
        freshSceneNames.add(pair.scene);
        freshResultRoots.set(pair.scene, "poc_data/fresh_generation_six_view");
      }
    }

    const originalGroup = document.createElement("optgroup");
    originalGroup.label = "Original POC scenes";
    for (const result of summary.scenes) {
      const option = document.createElement("option");
      option.value = result.scene;
      option.textContent = prettyName(result.scene);
      originalGroup.appendChild(option);
      sceneRoots.set(result.scene, `poc_data/${result.scene}`);
    }
    sceneSelect.appendChild(originalGroup);

    if (diverseSplitResponse.ok) {
      const split = await diverseSplitResponse.json();
      const addPairGroup = (label, sceneNames) => {
        const group = document.createElement("optgroup");
        group.label = label;
        for (const sceneName of sceneNames) {
          const option = document.createElement("option");
          option.value = sceneName;
          option.textContent = prettyName(sceneName);
          group.appendChild(option);
          pairSceneNames.add(sceneName);
          sceneRoots.set(sceneName, `poc_data/diverse_train/${sceneName}`);
        }
        sceneSelect.appendChild(group);
      };
      addPairGroup("Training scenes", split.splits.train);
      addPairGroup("Validation scenes", split.splits.validation);
    }

    if (heldoutResponse.ok) {
      const heldoutSummary = await heldoutResponse.json();
      const heldoutScenes = [...new Set(heldoutSummary.pairs.map(pair => pair.scene))];
      const heldoutGroup = document.createElement("optgroup");
      heldoutGroup.label = "Held-out scenes";
      for (const sceneName of heldoutScenes) {
        const option = document.createElement("option");
        option.value = sceneName;
        option.textContent = prettyName(sceneName);
        heldoutGroup.appendChild(option);
        pairSceneNames.add(sceneName);
        freshSceneNames.add(sceneName);
        sceneRoots.set(sceneName, `poc_data/heldout_alpha/${sceneName}`);
        freshResultRoots.set(
          sceneName, "poc_data/heldout_alpha_generation_six_view"
        );
      }
      sceneSelect.appendChild(heldoutGroup);
    }

    if (diverseTestResponse.ok) {
      const diverseTestSummary = await diverseTestResponse.json();
      const testScenes = [...new Set(diverseTestSummary.pairs.map(pair => pair.scene))];
      const testGroup = document.createElement("optgroup");
      testGroup.label = "Final untouched test";
      for (const sceneName of testScenes) {
        const option = document.createElement("option");
        option.value = sceneName;
        option.textContent = prettyName(sceneName);
        testGroup.appendChild(option);
        pairSceneNames.add(sceneName);
        freshSceneNames.add(sceneName);
        sceneRoots.set(sceneName, `poc_data/diverse_train/${sceneName}`);
        freshResultRoots.set(sceneName, "poc_data/diverse_test_final");
      }
      sceneSelect.appendChild(testGroup);
    }
    updateResultControls();
    await loadScene(summary.scenes[0].scene);
  } catch (error) {
    console.error(error);
    status.hidden = true;
    errorBox.textContent = error.message || "Failed to initialize viewer";
    errorBox.hidden = false;
  }
}

sceneSelect.addEventListener("change", () => {
  updateResultControls();
  loadScene(sceneSelect.value);
});
variantSelect.addEventListener("change", () => {
  updateResultControls();
  loadScene(sceneSelect.value);
});
seedSelect.addEventListener("change", () => loadScene(sceneSelect.value));
splatMode.addEventListener("change", updateRepresentation);
document.querySelector("#reset-camera").addEventListener("click", resetCamera);
new ResizeObserver(updateProjection).observe(viewer);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.autoClear = false;
  renderer.clear();
  renderer.render(scene, camera);
  renderer.clearDepth();
  renderer.render(overlayScene, camera);
});

resetCamera();
updateRepresentation();
initialize();
