/* Single-page flow: Upload -> Processing (poll /api/status) -> Chat.
   Plain JS, no dependencies. */

const $ = (id) => document.getElementById(id);

const views = { upload: $("view-upload"), processing: $("view-processing"), chat: $("view-chat") };

// fetch + JSON with friendly errors: a proxy/tunnel can return empty or
// non-JSON bodies (e.g. while the server restarts), which would otherwise
// surface as a raw "Unexpected end of JSON input" to the user.
async function fetchJSON(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error("Couldn't reach the server. Please try again in a moment.");
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok || data === null) {
    throw new Error((data && data.error) ||
      "The server didn't respond properly. Please try again in a moment.");
  }
  return data;
}
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
    const data = await fetchJSON("/api/process", { method: "POST", body: form });
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
    const data = await fetchJSON("/api/sample");
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
          const d = await fetchJSON("/api/process-sample", { method: "POST" });
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
  pollFailures = 0;
  $("process-error").hidden = true;
  document.querySelector(".process-card").hidden = false;
  STAGE_ORDER.forEach((s) => $("stage-" + s).classList.remove("active", "done"));
  $("process-message").textContent = "Starting…";
  showView("processing");
  pollTimer = setInterval(pollStatus, 1500);
  pollStatus();
}

let pollFailures = 0;

async function pollStatus() {
  if (pollInFlight || jobFinished) return;
  pollInFlight = true;
  let status;
  try {
    status = await fetchJSON(`/api/status/${jobId}`);
    pollFailures = 0;
  } catch (err) {
    // Tolerate transient blips (tunnel hiccups, brief server pauses):
    // only give up after several consecutive failures.
    pollFailures += 1;
    if (pollFailures >= 5) {
      jobFinished = true;
      stopPolling();
      showProcessingError(err.message || "Lost the connection to the server.");
    }
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

let articlesById = {};

function enterChat(status) {
  const n = status.article_count;
  $("article-count").textContent = `${n} article${n === 1 ? "" : "s"} found`;
  articlesById = {};
  const list = $("article-list");
  list.innerHTML = "";
  for (const a of status.articles) {
    articlesById[a.article_id] = a;
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = a.article_id + ".";
    btn.appendChild(num);
    btn.appendChild(document.createTextNode(a.title));
    btn.addEventListener("click", () => openArticle(a.article_id));
    li.appendChild(btn);
    list.appendChild(li);
  }
  $("messages").innerHTML = "";   // idempotent: never stack duplicate welcomes
  showView("chat");
  addBotMessage(
    `I've read all ${n} article${n === 1 ? "" : "s"} from your newspaper. ` +
    `Ask me anything about them — for example, "What are the main stories?"`);
  $("chat-input").focus();
}

/* --- article reader modal --- */

function openArticle(id) {
  const a = articlesById[id];
  if (!a) return;
  $("modal-title").textContent = a.title;
  $("modal-meta").textContent =
    `Article ${a.article_id}` + (a.source_page ? ` · from page ${a.source_page}` : "");
  $("modal-body").textContent = a.content;
  $("article-modal").hidden = false;
}

function closeArticle() { $("article-modal").hidden = true; }

$("modal-close").addEventListener("click", closeArticle);
$("article-modal").addEventListener("click", (e) => {
  if (e.target === $("article-modal")) closeArticle();  // click on backdrop
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("article-modal").hidden) closeArticle();
});

/* --- messages --- */

function addMessage(role, text, scrollToBottom = true) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  if (scrollToBottom) $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

const addBotMessage = (t, scroll = true) => addMessage("bot", t, scroll);

function addSources(msgEl, sources) {
  if (!sources || !sources.length) return;
  const div = document.createElement("div");
  div.className = "sources";
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = "Sources:";
  div.appendChild(label);
  for (const s of sources) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = s.title;
    chip.title = `Article ${s.article_id} — similarity ${s.score}. Click to read.`;
    chip.addEventListener("click", () => openArticle(s.article_id));
    div.appendChild(chip);
  }
  msgEl.appendChild(div);
}

const chatInput = $("chat-input");

// Chatbot-style composer: Enter sends, Shift+Enter makes a new line,
// and the box grows with the draft up to a cap.
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("chat-form").requestSubmit();
  }
});
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 224) + "px";
});

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question || !jobId) return;

  chatInput.value = "";
  chatInput.style.height = "";
  chatInput.disabled = true;
  $("btn-send").disabled = true;
  addMessage("user", question);
  const pending = addMessage("bot pending", "Reading the articles…");

  try {
    const data = await fetchJSON(`/api/chat/${jobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    pending.remove();
    // Don't yank the view to the bottom: anchor the top of the answer where
    // the pending bubble was, so the user reads from the start of the reply.
    const msgEl = addBotMessage(data.answer, false);
    addSources(msgEl, data.sources);
    const box = $("messages");
    box.scrollTop = Math.max(0, msgEl.offsetTop - 16);
  } catch (err) {
    pending.remove();
    addMessage("error", err.message || "Something went wrong — please try again.");
  } finally {
    chatInput.disabled = false;
    $("btn-send").disabled = false;
    chatInput.focus();
  }
});
