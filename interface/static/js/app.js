/* WebUI front-end logic */
(() => {
  const $ = (id) => document.getElementById(id);

  /* ================= slow playback (fixed fps, no loop) ================= */
  function startSlow(video, fps) {
    if (!video) return;
    fps = fps || 4;
    video.loop = false;
    video.pause();
    const step = 1 / fps;
    if (video.__slowTimer) { clearInterval(video.__slowTimer); video.__slowTimer = null; }
    const begin = () => {
      video.currentTime = 0;
      video.__slowTimer = setInterval(() => {
        if (video.currentTime + step >= (video.duration || Infinity)) {
          clearInterval(video.__slowTimer);
          video.__slowTimer = null;
          try { video.currentTime = video.duration; } catch (e) {}
          video.pause();
          return;
        }
        try { video.currentTime += step; } catch (e) {}
      }, Math.round(1000 / fps));
    };
    if (video.readyState >= 1) begin();
    else video.addEventListener("loadedmetadata", begin, { once: true });
  }
  window.startSlow = startSlow;

  /* ================= state ================= */
  // --- two independent upload areas (the canonical model inputs) ---
  let controlVideoFile = null;   // -> control_video
  let bgUploadFile = null;       // -> bg_image (Background Image area)
  let instanceRefFiles = [];     // -> subject_ref_images (Instance Reference Images area)

  // --- Scene Composer (parallel, optional; its composite = extra ref) ---
  let composerBgImg = null;      // HTMLImageElement drawn on the canvas
  let composerBgFile = null;     // the background file chosen inside the composer
  let composerSprites = [];      // {img, url, w, h, x, y}
  let composerSelected = null;

  let dragState = null;
  let polling = null;

  /* ================= dropzones (single-file) ================= */
  function setupDropzone(dzId, inputId, kind, onFile, onClear) {
    const dz = $(dzId), input = $(inputId);
    const empty = dz.querySelector(".dz-empty");
    const preview = dz.querySelector(".dz-preview");
    const media = preview.querySelector(kind === "video" ? "video" : "img");
    const clearBtn = preview.querySelector(".dz-clear");

    const show = (file) => {
      const url = URL.createObjectURL(file);
      media.src = url;
      if (kind === "video") startSlow(media, 4);
      empty.classList.add("hidden");
      preview.classList.remove("hidden");
      onFile(file);
    };
    dz.addEventListener("click", (e) => {
      if (e.target === clearBtn) return;
      input.click();
    });
    input.addEventListener("change", () => input.files[0] && show(input.files[0]));
    ["dragover", "dragenter"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
    dz.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (f) show(f);
    });
    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      media.src = "";
      input.value = "";
      preview.classList.add("hidden");
      empty.classList.remove("hidden");
      onClear();
    });

    dz.__showFile = show;   // exposed for demo auto-fill
  }

  setupDropzone("dzVideo", "fileVideo", "video",
    (f) => (controlVideoFile = f), () => (controlVideoFile = null));
  setupDropzone("dzBg", "fileBg", "image",
    (f) => (bgUploadFile = f), () => (bgUploadFile = null));

  /* ================= Instance Reference Images (independent, multi) ================= */
  const instRefRow = $("instRefRow"), instRefAdd = $("instRefAdd"), fileRefs = $("fileRefs");
  instRefAdd.addEventListener("click", () => fileRefs.click());
  fileRefs.addEventListener("change", () => {
    [...fileRefs.files].forEach(addInstanceRef);
    fileRefs.value = "";
  });
  instRefRow.addEventListener("dragover", (e) => e.preventDefault());
  instRefRow.addEventListener("drop", (e) => {
    e.preventDefault();
    [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")).forEach(addInstanceRef);
  });

  function addInstanceRef(file) {
    const url = URL.createObjectURL(file);
    instanceRefFiles.push({ file, url });
    renderInstanceRefs();
  }
  function renderInstanceRefs() {
    instRefRow.querySelectorAll(".ref-item").forEach((n) => n.remove());
    instanceRefFiles.forEach((r, i) => {
      const div = document.createElement("div");
      div.className = "ref-item";
      div.innerHTML = `<img src="${r.url}"/><span class="tag">Instance ${i + 1}</span><button class="dz-clear">✕</button>`;
      div.querySelector("button").addEventListener("click", () => {
        const idx = instanceRefFiles.indexOf(r);
        if (idx >= 0) instanceRefFiles.splice(idx, 1);
        renderInstanceRefs();
      });
      instRefRow.insertBefore(div, instRefAdd);
    });
  }
  renderInstanceRefs();

  /* ================= Scene Composer (canvas, parallel & optional) ================= */
  const canvas = $("composer"), ctx = canvas.getContext("2d");

  function syncCanvasSize() {
    const W = +$("width").value || 512, H = +$("height").value || 320;
    const maxSide = 640;
    const scale = Math.min(1, maxSide / Math.max(W, H));
    const nw = Math.max(64, Math.round(W * scale)), nh = Math.max(64, Math.round(H * scale));
    const ow = canvas.width || nw, oh = canvas.height || nh;
    canvas.width = nw; canvas.height = nh;
    const fx = nw / ow, fy = nh / oh;
    composerSprites.forEach((s) => { s.x *= fx; s.y *= fy; s.w *= fx; s.h *= fy; });
    drawComposer();
  }

  function drawComposer() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0d0f1c";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (composerBgImg) {
      const r = Math.max(canvas.width / composerBgImg.width, canvas.height / composerBgImg.height);
      const w = composerBgImg.width * r, h = composerBgImg.height * r;
      ctx.drawImage(composerBgImg, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    }
    composerSprites.forEach((s) => {
      ctx.drawImage(s.img, s.x - s.w / 2, s.y - s.h / 2, s.w, s.h);
      if (s === composerSelected) {
        ctx.save();
        ctx.strokeStyle = "#8b7dff"; ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(s.x - s.w / 2, s.y - s.h / 2, s.w, s.h);
        ctx.restore();
      }
    });
    const empty = !composerBgImg && composerSprites.length === 0;
    $("composerHint").classList.toggle("hidden", !empty);
  }

  function canvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  }
  function hitTest(px, py) {
    for (let i = composerSprites.length - 1; i >= 0; i--) {
      const s = composerSprites[i];
      if (px >= s.x - s.w / 2 && px <= s.x + s.w / 2 && py >= s.y - s.h / 2 && py <= s.y + s.h / 2) return s;
    }
    return null;
  }

  canvas.addEventListener("mousedown", (e) => {
    const p = canvasPos(e);
    const s = hitTest(p.x, p.y);
    if (s) {
      composerSelected = s;
      dragState = { s, dx: p.x - s.x, dy: p.y - s.y };
      composerSprites = composerSprites.filter((x) => x !== s); composerSprites.push(s);  // bring to front
    } else {
      composerSelected = null;
    }
    drawComposer();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragState) return;
    const p = canvasPos(e);
    dragState.s.x = p.x - dragState.dx;
    dragState.s.y = p.y - dragState.dy;
    drawComposer();
  });
  window.addEventListener("mouseup", () => { dragState = null; });
  canvas.addEventListener("wheel", (e) => {
    if (!composerSelected) return;
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.08 : 0.92;
    composerSelected.w *= f; composerSelected.h *= f;
    drawComposer();
  }, { passive: false });

  function setComposerBgFromFile(file) {
    composerBgFile = file;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { composerBgImg = img; drawComposer(); };
    img.src = url;
  }
  function addComposerInstance(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const side = Math.min(canvas.width, canvas.height) * 0.35;
      const ar = (img.width / img.height) || 1;
      let w = side, h = side;
      if (ar >= 1) h = side / ar; else w = side * ar;
      const sp = { img, url, w, h, x: canvas.width / 2, y: canvas.height / 2 };
      composerSprites.push(sp);
      composerSelected = sp;
      renderComposerRefs();
      drawComposer();
    };
    img.src = url;
  }

  $("btnBg").addEventListener("click", () => $("fileBgCanvas").click());
  $("fileBgCanvas").addEventListener("change", () => {
    if ($("fileBgCanvas").files[0]) setComposerBgFromFile($("fileBgCanvas").files[0]);
    $("fileBgCanvas").value = "";
  });
  $("btnAddInstComposer").addEventListener("click", () => $("fileAddInstComposer").click());
  $("fileAddInstComposer").addEventListener("change", () => {
    [...$("fileAddInstComposer").files].forEach(addComposerInstance);
    $("fileAddInstComposer").value = "";
  });
  $("btnClearInst").addEventListener("click", () => {
    composerSprites = []; composerSelected = null; renderComposerRefs(); drawComposer();
  });
  $("btnClearAll").addEventListener("click", () => {
    composerBgImg = null; composerBgFile = null;
    composerSprites = []; composerSelected = null;
    renderComposerRefs(); drawComposer();
  });

  function renderComposerRefs() {
    const row = $("composerRefRow");
    row.querySelectorAll(".ref-item").forEach((n) => n.remove());
    composerSprites.forEach((s, i) => {
      const div = document.createElement("div");
      div.className = "ref-item";
      div.innerHTML = `<img src="${s.url}"/><span class="tag">Instance ${i + 1}</span><button class="dz-clear">✕</button>`;
      div.querySelector("button").addEventListener("click", () => {
        const si = composerSprites.indexOf(s);
        if (si >= 0) {
          if (composerSelected === s) composerSelected = null;
          composerSprites.splice(si, 1);
        }
        renderComposerRefs();
        drawComposer();
      });
      row.appendChild(div);
    });
  }

  syncCanvasSize();

  /* ================= sliders ================= */
  function bindSlider(id, outId, fmt) {
    const el = $(id), out = $(outId);
    const update = () => {
      out.textContent = fmt ? fmt(+el.value) : el.value;
      const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
      el.style.setProperty("--fill", pct + "%");
    };
    el.addEventListener("input", update);
    update();
    return el;
  }
  const f2 = (v) => v.toFixed(2);
  const f1 = (v) => v.toFixed(1);

  bindSlider("videoLength", "videoLengthVal");
  bindSlider("fps", "fpsVal");
  bindSlider("steps", "stepsVal");
  bindSlider("cfg", "cfgVal", f1);
  bindSlider("textStrength", "textStrengthVal", f2);
  bindSlider("bgStrength", "bgStrengthVal", f2);
  bindSlider("insStrength", "insStrengthVal", f2);
  bindSlider("lora", "loraVal", f2);
  bindSlider("cfgSkip", "cfgSkipVal", f2);

  const updateDuration = () => {
    const n = +$("videoLength").value, fps = +$("fps").value;
    $("durationNote").textContent = `${n} frames ≈ ${(n / fps).toFixed(1)}s @${fps}fps`;
  };
  $("videoLength").addEventListener("input", updateDuration);
  $("fps").addEventListener("input", updateDuration);
  updateDuration();

  /* ================= resolution presets ================= */
  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("width").value = btn.dataset.w;
      $("height").value = btn.dataset.h;
      $("shift").value = +btn.dataset.h >= 720 || +btn.dataset.w >= 1280 ? 5 : 3;
      syncCanvasSize();
    });
  });
  ["width", "height"].forEach((id) =>
    $(id).addEventListener("input", () => {
      document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
      syncCanvasSize();
    }));

  /* ================= seed ================= */
  $("diceBtn").addEventListener("click", () => {
    $("seed").value = Math.floor(Math.random() * 2 ** 31);
  });

  /* ================= model status ================= */
  async function refreshModelStatus() {
    try {
      const r = await fetch("/api/model_status");
      const d = await r.json();
      const pill = $("modelStatusPill"), text = $("modelStatusText");
      pill.classList.remove("ready", "loading", "error");
      if (d.mock) { text.textContent = "Mock mode · no model needed"; pill.classList.add("ready"); return; }
      if (d.status === "ready") { text.textContent = "Model ready"; pill.classList.add("ready"); }
      else if (d.status === "loading") { text.textContent = d.message || "Loading model…"; pill.classList.add("loading"); }
      else if (d.status === "error") { text.textContent = "Model failed to load"; pill.classList.add("error"); }
      else text.textContent = "Model not loaded (auto-loads on first run)";
    } catch (e) { /* ignore */ }
  }
  refreshModelStatus();
  setInterval(refreshModelStatus, 5000);

  /* ================= generate ================= */
  const genBtn = $("generateBtn");
  const progressWrap = $("progressWrap"), progressFill = $("progressFill");
  const progressStage = $("progressStage"), progressPct = $("progressPct");
  const errorMsg = $("errorMsg");

  genBtn.addEventListener("click", () => {
    errorMsg.classList.add("hidden");
    if (!controlVideoFile) return showError("Please upload a sketch control video first");
    if (!bgUploadFile) return showError("Please set a Background Image first");
    if (instanceRefFiles.length === 0) return showError("Please add at least one Instance Reference Image");
    if (!$("prompt").value.trim()) return showError("Please enter a prompt");

    const fd = new FormData();
    fd.append("control_video", controlVideoFile);
    fd.append("bg_image", bgUploadFile);                       // dedicated Background Image area
    instanceRefFiles.forEach((r) => fd.append("subject_ref_images", r.file));  // dedicated Instance area

    fd.append("prompt", $("prompt").value.trim());
    fd.append("negative_prompt", $("negativePrompt").value.trim());
    fd.append("width", $("width").value);
    fd.append("height", $("height").value);
    fd.append("video_length", $("videoLength").value);
    fd.append("fps", $("fps").value);
    fd.append("num_inference_steps", $("steps").value);
    fd.append("guidance_scale", $("cfg").value);
    fd.append("seed", $("seed").value);
    fd.append("sampler_name", $("sampler").value);
    fd.append("shift", $("shift").value);
    fd.append("lora_weight", $("lora").value);
    fd.append("text_control_strength", $("textStrength").value);
    fd.append("bg_control_strength", $("bgStrength").value);
    fd.append("ins_control_strength", $("insStrength").value);
    fd.append("cfg_skip_ratio", $("cfgSkip").value);
    fd.append("enable_riflex", $("riflex").checked);

    genBtn.disabled = true;
    genBtn.querySelector(".gb-label").textContent = "⏳ Submitting…";
    setProgress(0, "Uploading input files…");
    progressWrap.classList.remove("hidden");

    // The Scene Composer composite (if any) is sent as an extra reference image
    // alongside the dedicated instance references — it never replaces them.
    const send = () => submitGen(fd);
    if (composerBgImg || composerSprites.length) {
      canvas.toBlob((blob) => {
        if (blob) fd.append("canvas_image", new File([blob], "scene_composer.png", { type: "image/png" }));
        send();
      }, "image/png");
    } else {
      send();
    }
  });

  function submitGen(fd) {
    fetch("/api/generate", { method: "POST", body: fd })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) throw new Error(d.error || "Failed to submit");
        pollStatus(d.job_id);
      })
      .catch((e) => { showError(e.message); resetBtn(); });
  }

  function pollStatus(jobId) {
    clearInterval(polling);
    polling = setInterval(async () => {
      try {
        const r = await fetch(`/api/status/${jobId}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || "Failed to query status");

        if (d.status === "queued") {
          setProgress(0.01, `Queued (position ${d.queue_position})…`);
        } else {
          setProgress(d.progress, d.stage);
        }

        if (d.status === "done") {
          clearInterval(polling);
          setProgress(1, "Done ✓");
          showResult(d.result_url, d.seed);
          resetBtn();
          setTimeout(() => progressWrap.classList.add("hidden"), 2500);
        } else if (d.status === "error") {
          clearInterval(polling);
          showError(d.error || "Generation failed");
          resetBtn();
        }
      } catch (e) {
        clearInterval(polling);
        showError(e.message);
        resetBtn();
      }
    }, 1000);
  }

  function setProgress(p, stage) {
    progressFill.style.width = Math.round(p * 100) + "%";
    progressPct.textContent = Math.round(p * 100) + "%";
    progressStage.textContent = stage;
  }
  function showError(msg) {
    errorMsg.textContent = "⚠ " + msg;
    errorMsg.classList.remove("hidden");
  }
  function resetBtn() {
    genBtn.disabled = false;
    genBtn.querySelector(".gb-label").textContent = "✦ Generate";
  }

  /* ================= result & history ================= */
  const historyRow = $("historyRow");
  function showResult(url, seed) {
    $("resultEmpty").classList.add("hidden");
    $("resultVideoWrap").classList.remove("hidden");
    const v = $("resultVideo");
    v.src = url;
    startSlow(v, 4);
    $("downloadBtn").href = url;
    $("resultSeed").textContent = seed != null ? `seed: ${seed}` : "";
    addHistory(url, seed);
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function addHistory(url, seed) {
    const item = document.createElement("div");
    item.className = "h-item";
    item.innerHTML = `<video src="${url}" muted loop playsinline></video>`;
    item.title = seed != null ? `seed: ${seed}` : "";
    item.addEventListener("mouseenter", () => item.querySelector("video").play().catch(() => {}));
    item.addEventListener("mouseleave", () => item.querySelector("video").pause());
    item.addEventListener("click", () => {
      historyRow.querySelectorAll(".h-item").forEach((n) => n.classList.remove("active"));
      item.classList.add("active");
      const v = $("resultVideo");
      v.src = url;
      startSlow(v, 4);
      $("downloadBtn").href = url;
      $("resultSeed").textContent = item.title;
    });
    historyRow.querySelectorAll(".h-item").forEach((n) => n.classList.remove("active"));
    item.classList.add("active");
    historyRow.prepend(item);
  }

  /* ================= demo auto-fill ================= */
  async function urlToFile(url, filename) {
    const resp = await fetch(url);
    const blob = await resp.blob();
    return new File([blob], filename, { type: blob.type });
  }

  async function prefillDemo() {
    try {
      const r = await fetch("/api/demo");
      const d = await r.json();
      if (!d.available) return;

      if (d.sketch) {
        const f = await urlToFile(d.sketch, d.sketch.split("/").pop());
        $("dzVideo").__showFile(f);
      }
      if (d.bg) {
        const f = await urlToFile(d.bg, d.bg.split("/").pop());
        $("dzBg").__showFile(f);            // dedicated Background Image area
        setComposerBgFromFile(f);           // also seed the composer canvas
      }
      for (const u of (d.refs || [])) {
        const f = await urlToFile(u, u.split("/").pop());
        addInstanceRef(f);                  // dedicated Instance Reference Images
        addComposerInstance(f);            // also place into the composer
      }
      if (d.prompt && !$("prompt").value.trim()) {
        $("prompt").value = d.prompt;
      }
      if (d.output) {
        showResult(d.output, null);
      }
    } catch (e) {
      /* demo prefill is best-effort; ignore failures */
    }
  }
  prefillDemo();
})();
