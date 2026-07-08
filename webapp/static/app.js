/* Single-page flow: Upload -> Processing (poll /api/status) -> Chat.
   Plain JS, no dependencies. */

const $ = (id) => document.getElementById(id);

const views = { upload: $("view-upload"), processing: $("view-processing"), chat: $("view-chat") };
let jobId = null;
let pollTimer = null;
let pollInFlight = false;   // don't stack overlapping status requests
let jobFinished = false;    // ensures ready/error is handled exactly once
let startingJob = false;    // ignore extra clicks while a job is being started

/* ---------- view switching ---------- */

function showView(name) {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== name;
  document.querySelectorAll("#steps li").forEach((li) => {
    const order = ["upload", "processing", "chat"];
    const here = order.indexOf(li.dataset.step);
    const now = order.indexOf(name);
    li.classList.toggle("current", here === now);
    li.classList.toggle("done", here < now);
  });
}

/* ---------- stage 1: upload ---------- */

const dropzone = $("dropzone");
const fileInput = $("file-input");

$("btn-browse").addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", (e) => {
  if (e.target === dropzone || e.target.closest(".dz-icon, .dz-title, .dz-sub")) fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

function showUploadError(msg) {
  const el = $("upload-error");
  el.textContent = msg;
  el.hidden = false;
}

async function uploadFile(file) {
  if (startingJob) return;
  $("upload-error").hidden = true;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showUploadError("That doesn't look like a PDF. Please choose a .pdf file.");
    return;
  }
  startingJob = true;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/process", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed.");
    beginProcessing(data.job_id);
  } catch (err) {
    showUploadError(err.message || "Could not upload the file. Please try again.");
  } finally {
    startingJob = false;
    fileInput.value = "";
  }
}

// Offer the example PDF only if the server actually has one installed.
(async function initSample() {
  try {
    const res = await fetch("/api/sample");
    const data = await res.json();
    if (data.available) {
      $("sample-offer").hidden = false;
      $("btn-sample").addEventListener("click", async () => {
        if (startingJob) return;
        startingJob = true;
        const btn = $("btn-sample");
        btn.disabled = true;
        btn.textContent = "Starting…";   // immediate feedback on click
        $("upload-error").hidden = true;
        try {
          const r = await fetch("/api/process-sample", { method: "POST" });
          const d = await r.json();
          if (!r.ok) throw new Error(d.error || "Could not start the example.");
          beginProcessing(d.job_id);
        } catch (err) {
          showUploadError(err.message);
        } finally {
          startingJob = false;
          btn.disabled = false;
          btn.textContent = "Try the example newspaper";
        }
      });
    }
  } catch { /* sample stays hidden */ }
})();

/* ---------- stage 2: processing ---------- */

const STAGE_ORDER = ["ocr", "segmenting", "embedding"];

function beginProcessing(id) {
  stopPolling();          // never leak an interval from a previous job
  jobId = id;
  jobFinished = false;
  pollInFlight = false;
  $("process-error").hidden = true;
  document.querySelector(".process-card").hidden = false;
  STAGE_ORDER.forEach((s) => $("stage-" + s).classList.remove("active", "done"));
  $("process-message").textContent = "Starting…";
  showView("processing");
  pollTimer = setInterval(pollStatus, 1500);
  pollStatus();
}

async function pollStatus() {
  if (pollInFlight || jobFinished) return;
  pollInFlight = true;
  let status;
  try {
    const res = await fetch(`/api/status/${jobId}`);
    status = await res.json();
    if (!res.ok) throw new Error(status.error || "Lost track of this job.");
  } catch (err) {
    jobFinished = true;
    stopPolling();
    showProcessingError(err.message || "Lost the connection to the server.");
    return;
  } finally {
    pollInFlight = false;
  }
  if (jobFinished) return;

  $("process-message").textContent = status.message || "";

  const idx = STAGE_ORDER.indexOf(status.stage);
  STAGE_ORDER.forEach((s, i) => {
    const li = $("stage-" + s);
    li.classList.toggle("done", idx > i || status.stage === "ready");
    li.classList.toggle("active", idx === i);
  });

  if (status.stage === "error") {
    jobFinished = true;
    stopPolling();
    showProcessingError(status.error || "Something went wrong.");
  } else if (status.stage === "ready") {
    jobFinished = true;
    stopPolling();
    enterChat(status);
  }
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

function showProcessingError(msg) {
  document.querySelector(".process-card").hidden = true;
  $("process-error-text").textContent = msg;
  $("process-error").hidden = false;
}

$("btn-retry").addEventListener("click", resetToUpload);
$("btn-newdoc").addEventListener("click", resetToUpload);

function resetToUpload() {
  stopPolling();
  jobId = null;
  $("messages").innerHTML = "";
  $("upload-error").hidden = true;
  showView("upload");
}

/* ---------- stage 3: chat ---------- */

function enterChat(status) {
  const n = status.article_count;
  $("article-count").textContent = `${n} article${n === 1 ? "" : "s"} found`;
  const list = $("article-list");
  list.innerHTML = "";
  for (const a of status.articles) {
    const li = document.createElement("li");
    li.textContent = a.title;
    list.appendChild(li);
  }
  $("messages").innerHTML = "";   // idempotent: never stack duplicate welcomes
  showView("chat");
  addBotMessage(
    `I've read all ${n} article${n === 1 ? "" : "s"} from your newspaper. ` +
    `Ask me anything about them — for example, "What are the main stories?"`);
  $("chat-input").focus();
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

const addBotMessage = (t) => addMessage("bot", t);

function addSources(msgEl, sources) {
  if (!sources || !sources.length) return;
  const div = document.createElement("div");
  div.className = "sources";
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = "Sources:";
  div.appendChild(label);
  for (const s of sources) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = s.title;
    chip.title = `Article ${s.article_id} — similarity ${s.score}`;
    div.appendChild(chip);
  }
  msgEl.appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const question = input.value.trim();
  if (!question || !jobId) return;

  input.value = "";
  input.disabled = true;
  $("btn-send").disabled = true;
  addMessage("user", question);
  const pending = addMessage("bot pending", "Reading the articles…");

  try {
    const res = await fetch(`/api/chat/${jobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "The model didn't respond.");
    pending.remove();
    const msgEl = addBotMessage(data.answer);
    addSources(msgEl, data.sources);
  } catch (err) {
    pending.remove();
    addMessage("error", err.message || "Something went wrong — please try again.");
  } finally {
    input.disabled = false;
    $("btn-send").disabled = false;
    input.focus();
  }
});
