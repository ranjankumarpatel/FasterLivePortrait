const state = {
  apiBase: window.LIVE_AVATAR_CONFIG?.apiBase || "",
  sessionId: null,
  cameraStream: null,
  socket: null,
  frameTimer: null,
};

const els = {
  healthBadge: document.querySelector("#healthBadge"),
  readyBadge: document.querySelector("#readyBadge"),
  sessionForm: document.querySelector("#sessionForm"),
  sourceImage: document.querySelector("#sourceImage"),
  animalMode: document.querySelector("#animalMode"),
  sessionOutput: document.querySelector("#sessionOutput"),
  renderForm: document.querySelector("#renderForm"),
  drivingVideo: document.querySelector("#drivingVideo"),
  drivingPickle: document.querySelector("#drivingPickle"),
  renderOutput: document.querySelector("#renderOutput"),
  cameraPreview: document.querySelector("#cameraPreview"),
  avatarPreview: document.querySelector("#avatarPreview"),
  startCamera: document.querySelector("#startCamera"),
  startStream: document.querySelector("#startStream"),
  stopStream: document.querySelector("#stopStream"),
  streamOutput: document.querySelector("#streamOutput"),
};

function setBadge(element, text, mode) {
  element.textContent = text;
  element.classList.remove("ok", "warn");
  if (mode) element.classList.add(mode);
}

async function refreshStatus() {
  try {
    const health = await fetch(`${state.apiBase}/healthz`);
    const healthPayload = await health.json();
    setBadge(els.healthBadge, `health: ${healthPayload.status}`, "ok");
  } catch (error) {
    setBadge(els.healthBadge, "health: offline", "warn");
  }

  try {
    const ready = await fetch(`${state.apiBase}/readyz`);
    if (ready.ok) {
      setBadge(els.readyBadge, "engine: ready", "ok");
    } else {
      setBadge(els.readyBadge, "engine: lazy", "warn");
    }
  } catch (error) {
    setBadge(els.readyBadge, "engine: unknown", "warn");
  }
}

async function createSession(event) {
  event.preventDefault();
  if (!els.sourceImage.files.length) {
    els.sessionOutput.textContent = "Choose an avatar image.";
    return;
  }

  const form = new FormData();
  form.append("source_image", els.sourceImage.files[0]);
  form.append("animal", els.animalMode.checked ? "true" : "false");

  els.sessionOutput.textContent = "Creating session...";
  const response = await fetch(`${state.apiBase}/v1/avatar/sessions`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    els.sessionOutput.textContent = await response.text();
    return;
  }

  const payload = await response.json();
  state.sessionId = payload.id;
  els.sessionOutput.textContent = `Session ${payload.id}`;
}

async function renderAvatar(event) {
  event.preventDefault();
  if (!state.sessionId) {
    els.renderOutput.textContent = "Create a session first.";
    return;
  }

  const form = new FormData();
  if (els.drivingVideo.files.length) form.append("driving_video", els.drivingVideo.files[0]);
  if (els.drivingPickle.files.length) form.append("driving_pickle", els.drivingPickle.files[0]);

  els.renderOutput.textContent = "Rendering...";
  const response = await fetch(`${state.apiBase}/v1/avatar/sessions/${state.sessionId}/render`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    els.renderOutput.textContent = await response.text();
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "avatar-render.zip";
  anchor.click();
  URL.revokeObjectURL(url);
  els.renderOutput.textContent = "Downloaded avatar-render.zip";
}

async function startCamera() {
  state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  els.cameraPreview.srcObject = state.cameraStream;
  els.streamOutput.textContent = "Camera ready.";
}

function stopStream() {
  if (state.frameTimer) clearInterval(state.frameTimer);
  state.frameTimer = null;
  if (state.socket) state.socket.close();
  state.socket = null;
  els.streamOutput.textContent = "WebSocket: stopped.";
}

function makeFrameCanvas() {
  const canvas = document.createElement("canvas");
  const width = els.cameraPreview.videoWidth || 640;
  const height = els.cameraPreview.videoHeight || 360;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  context.drawImage(els.cameraPreview, 0, 0, width, height);
  return canvas;
}

async function startStream() {
  if (!state.sessionId) {
    els.streamOutput.textContent = "Create a session first.";
    return;
  }
  if (!state.cameraStream) await startCamera();

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socketUrl = `${protocol}://${window.location.host}/v1/avatar/sessions/${state.sessionId}/stream?output=org`;
  state.socket = new WebSocket(socketUrl);
  state.socket.binaryType = "blob";

  state.socket.onopen = () => {
    els.streamOutput.textContent = "WebSocket: streaming.";
    state.frameTimer = setInterval(() => {
      if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
      const canvas = makeFrameCanvas();
      canvas.toBlob((blob) => {
        if (blob && state.socket?.readyState === WebSocket.OPEN) state.socket.send(blob);
      }, "image/jpeg", 0.78);
    }, 160);
  };

  state.socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      els.streamOutput.textContent = event.data;
      return;
    }
    const oldUrl = els.avatarPreview.dataset.url;
    const url = URL.createObjectURL(event.data);
    els.avatarPreview.src = url;
    els.avatarPreview.dataset.url = url;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
  };

  state.socket.onerror = () => {
    els.streamOutput.textContent = "WebSocket: error.";
  };

  state.socket.onclose = () => {
    if (state.frameTimer) clearInterval(state.frameTimer);
    state.frameTimer = null;
    els.streamOutput.textContent = "WebSocket: closed.";
  };
}

els.sessionForm.addEventListener("submit", createSession);
els.renderForm.addEventListener("submit", renderAvatar);
els.startCamera.addEventListener("click", startCamera);
els.startStream.addEventListener("click", startStream);
els.stopStream.addEventListener("click", stopStream);

refreshStatus();
setInterval(refreshStatus, 5000);
