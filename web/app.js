const API = "/api";
const $ = id => document.getElementById(id);

let cameraStream = null;
let cameraTimer = null;
let cameraBusy = false;


// ─────────────────────────────────────────────
// SYSTEM HEALTH
// ─────────────────────────────────────────────

async function loadHealth() {
  try {
    const response = await fetch(`${API}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    const data = await response.json();

    console.log("HEALTH:", data);

    $("model").textContent =
      (data.model || "buffalo_l").toUpperCase();

    $("threshold").textContent =
      Number(data.threshold ?? 0.45).toFixed(2);

    $("identities").textContent =
      data.identities ?? "—";

    $("systemState").textContent =
      "SYSTEM ONLINE";

  } catch (error) {

    console.error("HEALTH CHECK ERROR:", error);

    $("systemState").textContent =
      "API UNAVAILABLE";
  }
}

loadHealth();


// ─────────────────────────────────────────────
// IMAGE IDENTIFICATION
// ─────────────────────────────────────────────

const dz = $("dropzone");
const input = $("identifyInput");

$("identifyBtn").onclick = () => input.click();

input.onchange = () => {
  if (input.files[0]) {
    identify(input.files[0]);
  }
};

["dragenter", "dragover"].forEach(event => {
  dz.addEventListener(event, e => {
    e.preventDefault();
    dz.classList.add("drag");
  });
});

["dragleave", "drop"].forEach(event => {
  dz.addEventListener(event, e => {
    e.preventDefault();
    dz.classList.remove("drag");
  });
});

dz.addEventListener("drop", e => {
  if (e.dataTransfer.files[0]) {
    identify(e.dataTransfer.files[0]);
  }
});


async function identify(file) {

  const preview = $("preview");

  preview.style.display = "block";

  preview.innerHTML = `
    <img src="${URL.createObjectURL(file)}" alt="query face">
  `;

  setProcessing();

  const fd = new FormData();
  fd.append("file", file);

  try {

    const r = await fetch(`${API}/identify`, {
      method: "POST",
      body: fd
    });

    const d = await r.json();

    if (!r.ok) {
      throw new Error(d.detail || "Request failed");
    }

    displayResult(d);

  } catch (e) {

    showError(e.message);
  }
}


// ─────────────────────────────────────────────
// CAMERA
// ─────────────────────────────────────────────

const startCameraBtn = $("startCameraBtn");
const stopCameraBtn = $("stopCameraBtn");
const cameraView = $("cameraView");
const cameraVideo = $("cameraVideo");
const cameraStatus = $("cameraStatus");


startCameraBtn.onclick = startCamera;
stopCameraBtn.onclick = stopCamera;


async function startCamera() {

  try {

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    cameraVideo.srcObject = cameraStream;

    cameraView.classList.add("active");

    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;

    cameraStatus.textContent = "CAMERA ACTIVE";

    $("resultIdentity").textContent = "Looking for face…";
    $("statusBadge").textContent = "SCANNING";
    $("statusBadge").className = "status-badge";
    $("resultScore").textContent = "—";
    $("meterFill").style.width = "0%";

    // Give the camera a moment to initialize.
    setTimeout(() => {

      cameraTimer = setInterval(
        captureCameraFrame,
        1200
      );

    }, 1000);

  } catch (error) {

    cameraStatus.textContent = "CAMERA ACCESS DENIED";

    showError(
      "Camera access is unavailable. Allow camera permission in your browser."
    );
  }
}
function stopCamera() {

  if (cameraTimer) {
    clearInterval(cameraTimer);
    cameraTimer = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(
      track => track.stop()
    );

    cameraStream = null;
  }

  cameraVideo.srcObject = null;

  cameraView.classList.remove("active");

  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;

  cameraStatus.textContent = "CAMERA READY";

  // Reset analysis result
  $("statusBadge").textContent = "WAITING";
  $("statusBadge").className = "status-badge";

  $("resultIdentity").textContent =
    "No image analysed";

  $("resultScore").textContent = "—";

  $("meterFill").style.width = "0%";

  $("resultReason").textContent =
    "Upload an image or start the camera.";

  $("resultThreshold").textContent =
    `Threshold ${Number(
      document.getElementById("threshold").textContent
    ).toFixed(2)}`;

  // Reset pipeline
  ["retinaNode", "arcfaceNode", "matchNode"]
    .forEach(id => {
      const node = $(id);
      if (node) {
        node.className = "node pipeline-node";
      }
    });
}

async function captureCameraFrame() {

  if (!cameraStream || cameraBusy) {
    return;
  }

  if (
    !cameraVideo.videoWidth ||
    !cameraVideo.videoHeight
  ) {
    return;
  }

  cameraBusy = true;

  try {

    const canvas = document.createElement("canvas");

    // Keep frames reasonably small for faster inference.
    const width = 640;
    const height =
      Math.round(
        cameraVideo.videoHeight *
        (width / cameraVideo.videoWidth)
      );

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(
      cameraVideo,
      0,
      0,
      width,
      height
    );

    canvas.toBlob(
      async blob => {

        if (!blob) {
          cameraBusy = false;
          return;
        }

        const file = new File(
          [blob],
          "camera-frame.jpg",
          { type: "image/jpeg" }
        );

        await identifyCameraFrame(file);

        cameraBusy = false;

      },
      "image/jpeg",
      0.85
    );

  } catch (error) {

    cameraBusy = false;
  }
}


async function identifyCameraFrame(file) {

  const fd = new FormData();

  fd.append("file", file);

  try {

    const r = await fetch(
      `${API}/camera/identify`,
      {
        method: "POST",
        body: fd
      }
    );

    const d = await r.json();

    if (!r.ok) {
      return;
    }

    displayResult(d);

    if (d.status === "MATCH") {
      cameraStatus.textContent = "IDENTITY CONFIRMED";
    }

    else if (d.status === "UNKNOWN") {
      cameraStatus.textContent = "UNKNOWN FACE";
    }

    else if (d.status === "NO_FACE") {
      cameraStatus.textContent = "SEARCHING FOR FACE";
    }

    else if (d.status === "MULTIPLE_FACES") {
      cameraStatus.textContent = "MULTIPLE FACES";
    }

  } catch (error) {

    cameraStatus.textContent = "PIPELINE ERROR";
  }
}


// ─────────────────────────────────────────────
// RESULT DISPLAY
// ─────────────────────────────────────────────

function setProcessing() {

  $("resultIdentity").textContent = "Analysing…";

  $("statusBadge").textContent = "PROCESSING";

  $("statusBadge").className =
    "status-badge";

  $("resultScore").textContent = "—";

  $("meterFill").style.width = "0%";
}

function updatePipeline(status) {

  const retina = $("retinaNode");
  const arcface = $("arcfaceNode");
  const matcher = $("matchNode");

  if (!retina || !arcface || !matcher) {
    return;
  }

  // Reset
  retina.className = "node pipeline-node";
  arcface.className = "node pipeline-node";
  matcher.className = "node pipeline-node";

  // Face successfully detected and embedding generated
  if (
    status === "MATCH" ||
    status === "UNKNOWN"
  ) {
    retina.classList.add("success");
    arcface.classList.add("success");
  }

  // Final matching stage
  if (status === "MATCH") {
    matcher.classList.add("success");
  }

  else if (status === "UNKNOWN") {
    matcher.classList.add("warning");
  }

  else if (status === "MULTIPLE_FACES") {
    retina.classList.add("warning");
  }

  else if (status === "NO_FACE") {
    retina.classList.add("warning");
  }

  else if (status === "ERROR") {
    retina.classList.add("error");
  }
}

function displayResult(d) {
  updatePipeline(d.status);
  const score = d.similarity;

  $("statusBadge").textContent =
    d.status;

  $("statusBadge").className =
    `status-badge ${
      d.status === "MATCH"
        ? "match"
        : d.status === "UNKNOWN"
          ? "unknown"
          : ""
    }`;

  $("resultIdentity").textContent =
    d.identity ||
    (
      d.status === "NO_FACE"
        ? "No face detected"
        : d.status === "MULTIPLE_FACES"
          ? "Multiple faces detected"
          : "Unknown face"
    );

  $("resultScore").textContent =
    score == null
      ? "—"
      : score.toFixed(3);

  $("meterFill").style.width =
    score == null
      ? "0%"
      : `${Math.max(
          0,
          Math.min(100, score * 100)
        )}%`;

  $("resultReason").textContent =
    d.reason ||
    "Similarity cleared the acceptance threshold.";

  $("resultThreshold").textContent =
    `Threshold ${Number(
      d.threshold
    ).toFixed(2)}`;
}


function showError(message) {

  $("statusBadge").textContent =
    "ERROR";

  $("statusBadge").className =
    "status-badge";

  $("resultIdentity").textContent =
    message;

  $("resultScore").textContent =
    "—";

  $("meterFill").style.width =
    "0%";

  $("resultReason").textContent =
    "Pipeline request failed.";
}


// ─────────────────────────────────────────────
// ENROLLMENT
// ─────────────────────────────────────────────

const enrollInput = $("enrollInput");

enrollInput.onchange = () => {

  $("fileList").innerHTML =
    [...enrollInput.files]
      .map(
        f => `<div>✓ ${f.name}</div>`
      )
      .join("");
};


$("enrollBtn").onclick =
  async () => {

    const identity =
      $("identity").value.trim();

    if (
      !identity ||
      !enrollInput.files.length
    ) {

      $("enrollStatus").textContent =
        "Enter an identity and select images.";

      return;
    }

    const fd = new FormData();

    fd.append(
      "identity",
      identity
    );

    [...enrollInput.files].forEach(
      file =>
        fd.append(
          "files",
          file
        )
    );

    $("enrollStatus").textContent =
      "Processing enrollment…";

    try {

      const r = await fetch(
        `${API}/enroll`,
        {
          method: "POST",
          body: fd
        }
      );

      const d = await r.json();

      if (!r.ok) {

        throw new Error(
          typeof d.detail === "string"
            ? d.detail
            : d.detail?.message ||
              "Enrollment failed"
        );
      }

      $("enrollStatus").textContent =
        `${d.identity} enrolled · ` +
        `${d.enrolled_images} valid image(s).`;

      loadHealth();

    } catch (e) {

      $("enrollStatus").textContent =
        e.message;
    }
  };


// Stop camera if the page is closed.
window.addEventListener(
  "beforeunload",
  stopCamera
);