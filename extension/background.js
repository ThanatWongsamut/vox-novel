// VoxNovel batch importer: the job driver.
//
// Loads each chapter of a job in one tab of the user's own Chrome, reads it
// through content.js and posts it to the VoxNovel server. The plan comes from
// the server, never from whoever asked for the job, and every URL in it is
// checked before a page is opened.
//
// Manifest V3 suspends this worker whenever it is idle, so nothing important
// lives in memory: the job is in chrome.storage.local and every step can be
// resumed from there by whichever event wakes the worker next.

importScripts("lib/core.js");
const C = self.VoxCore;

const JOB_KEY = "voxJob";
const TAB_KEY = "voxJobTabId";
const HEARTBEAT_ALARM = "vox-job-heartbeat";
const CUSTOM_RELAY_ID = "vox-relay-custom";
const PAGE_LOAD_TIMEOUT_MS = 45000;
const COMMANDS = new Set(["start", "pause", "resume", "skip", "cancel", "status", "clear"]);

// Only guards against two drivers in one worker. After a restart nothing is
// mid-step -- the step died with the old worker -- so starting fresh is right.
let driving = false;
const ports = new Set();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------- storage --

async function getServerUrl() {
  const { voxServerUrl } = await chrome.storage.local.get("voxServerUrl");
  return (voxServerUrl || C.DEFAULT_SERVER_URL).replace(/\/+$/, "");
}

async function trustedOrigins() {
  const origins = new Set(C.DEFAULT_SERVER_ORIGINS);
  const configured = C.serverOrigin(await getServerUrl());
  if (configured) origins.add(configured);
  return origins;
}

async function loadJob() {
  return (await chrome.storage.local.get(JOB_KEY))[JOB_KEY] || null;
}

async function saveJob(job) {
  await chrome.storage.local.set({ [JOB_KEY]: job });
  publish(job);
  await syncHeartbeat(job);
  if (job && job.status === "done") await closeJobTab();
}

async function dispatch(event) {
  const job = await loadJob();
  const next = C.reduce(job, event, Date.now());
  if (next !== job) await saveJob(next);
  return next;
}

// ---------------------------------------------------------- presentation --

function publish(job) {
  const message = { type: "job", job: C.summarize(job) };
  for (const port of ports) {
    try {
      port.postMessage(message);
    } catch (e) {
      ports.delete(port);
    }
  }
  const badge =
    !job || !C.isActive(job) ? "" : job.status === "paused" ? "!" : String(job.queue.length - job.index);
  chrome.action.setBadgeText({ text: badge });
  chrome.action.setBadgeBackgroundColor({ color: job && job.status === "paused" ? "#f59e0b" : "#6366f1" });
}

// A running job gets a heartbeat, so a worker suspended mid-job is woken to
// carry on. A paused one waits for the user and needs none.
async function syncHeartbeat(job) {
  if (job && job.status === "running") {
    await chrome.alarms.create(HEARTBEAT_ALARM, { periodInMinutes: 0.5 });
  } else {
    await chrome.alarms.clear(HEARTBEAT_ALARM);
  }
}

// ------------------------------------------------------------- the tab --

async function storedTabId() {
  return (await chrome.storage.local.get(TAB_KEY))[TAB_KEY];
}

async function existingTab() {
  const id = await storedTabId();
  if (id == null) return null;
  try {
    return await chrome.tabs.get(id);
  } catch (e) {
    return null;
  }
}

/** Load a URL in the job tab, creating it if needed, and wait for it to finish. */
async function navigateAndWait(url) {
  const tab = await existingTab();
  return new Promise((resolve, reject) => {
    let tabId = tab ? tab.id : null;
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("The page did not finish loading."));
    }, PAGE_LOAD_TIMEOUT_MS);

    // Registered before navigating, so the completion cannot be missed.
    function onUpdated(updatedId, info, updatedTab) {
      if (updatedId === tabId && info.status === "complete") {
        cleanup();
        resolve(updatedTab);
      }
    }
    function cleanup() {
      clearTimeout(timer);
      chrome.tabs.onUpdated.removeListener(onUpdated);
    }
    chrome.tabs.onUpdated.addListener(onUpdated);

    (async () => {
      if (tab) {
        await chrome.tabs.update(tabId, { url });
      } else {
        // In the background: the user only needs this tab when a job pauses.
        const created = await chrome.tabs.create({ url, active: false });
        tabId = created.id;
        await chrome.storage.local.set({ [TAB_KEY]: tabId });
      }
    })().catch((e) => {
      cleanup();
      reject(e);
    });
  });
}

/**
 * Close the job's tab once the job is done. Only while it still shows a
 * ReadToon chapter: if the user has since used the tab for something else, it
 * is theirs now. A cancelled job keeps its tab -- cancelling often happens
 * mid-login or mid-purchase on exactly that page.
 */
async function closeJobTab() {
  const tab = await existingTab();
  await chrome.storage.local.remove(TAB_KEY);
  if (tab && C.isReadtoonChapterUrl(tab.url)) {
    try {
      await chrome.tabs.remove(tab.id);
    } catch (e) {
      // Already closed.
    }
  }
}

async function bringJobTabForward() {
  const tab = await existingTab();
  if (!tab) return;
  await chrome.tabs.update(tab.id, { active: true });
  await chrome.windows.update(tab.windowId, { focused: true });
}

/** Ask content.js to wait for the chapter and report what it found. */
async function readPage(tabId) {
  // The content script can still be starting when the load completes.
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const reply = await chrome.tabs.sendMessage(tabId, {
        action: "EXTRACT_WHEN_READY",
        timeoutMs: C.EXTRACT_TIMEOUT_MS,
      });
      if (reply) return reply;
    } catch (e) {
      // Not listening yet.
    }
    await sleep(500);
  }
  return null;
}

// ---------------------------------------------------------- the server --

async function postImport(details) {
  let response;
  try {
    response = await fetch(`${await getServerUrl()}/api/extension/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(C.buildImportPayload(details)),
    });
  } catch (e) {
    return { type: "server", detail: String(e.message || e) };
  }
  if (response.ok) return { type: "imported" };
  const body = await response.json().catch(() => ({}));
  const detail = body.detail || `HTTP ${response.status}`;
  // A 5xx is the server's trouble and will not fix itself by moving on. A 4xx
  // is about this chapter's content, so it is retried and then skipped.
  return response.status >= 500 ? { type: "server", detail } : { type: "empty", detail };
}

async function fetchPlan({ seriesId, from, to, includeImported }) {
  const params = new URLSearchParams({ series_id: seriesId });
  if (from != null) params.set("from", String(from));
  if (to != null) params.set("to", String(to));
  // A re-fetch: chapters already in VoxNovel are loaded again. Their speaker
  // labels survive, since the server carries them over by paragraph text.
  if (includeImported) params.set("include_imported", "true");
  let response;
  try {
    response = await fetch(`${await getServerUrl()}/api/extension/plan?${params}`);
  } catch (e) {
    throw new Error("Could not reach the VoxNovel server. Is it running (uv run vox-novel web)?");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `The server refused the plan (HTTP ${response.status}).`);
  return body;
}

// ----------------------------------------------------------- the driver --

/** Load, read and import one chapter. Returns an outcome for the reducer. */
async function attempt(chapter) {
  let tab;
  try {
    tab = await navigateAndWait(chapter.url);
  } catch (e) {
    return { type: "empty", detail: String(e.message || e) };
  }
  // Without the "tabs" permission the URL is only visible on ReadToon, so a
  // redirect off the site reads as undefined -- which is a redirect too.
  if (!C.samePage(tab.url, chapter.url)) return { type: "redirected" };

  const reply = await readPage(tab.id);
  if (!reply || !reply.ok) return { type: "empty", detail: "Could not read the page." };
  if (reply.status !== "ok") return { type: reply.status };
  if (!C.samePage(reply.details.url, chapter.url)) return { type: "redirected" };
  return postImport(reply.details);
}

async function drive() {
  if (driving) return;
  driving = true;
  try {
    for (;;) {
      const job = await loadJob();
      if (!job || job.status !== "running") return;
      const chapter = C.current(job);
      if (!chapter) return;

      const outcome = await attempt(chapter);

      // The user may have paused, skipped or cancelled while the page loaded.
      // Then this outcome is stale; the loop re-reads the job and acts on that.
      const now = await loadJob();
      if (!now || now.id !== job.id || now.index !== job.index || now.status !== "running") continue;

      const next = await dispatch({ type: "outcome", outcome: outcome.type, detail: outcome.detail });
      if (next.status === "paused") {
        await bringJobTabForward();
        return;
      }
      if (next.status === "running") await sleep(C.CHAPTER_DELAY_MS);
    }
  } finally {
    driving = false;
  }
}

// ---------------------------------------------------------- commands --

async function isTrusted(sender) {
  if (sender.id !== chrome.runtime.id) return false;
  // The popup and other extension pages have no tab.
  if (!sender.tab) return true;
  // Otherwise only the relay, in the top frame of a VoxNovel page.
  if (sender.frameId !== 0) return false;
  return (await trustedOrigins()).has(C.serverOrigin(sender.url || sender.tab.url));
}

function readRange(payload) {
  const number = (v) => (v === null || v === undefined || v === "" ? null : Number(v));
  const from = number(payload.from);
  const to = number(payload.to);
  for (const v of [from, to]) {
    if (v !== null && !Number.isFinite(v)) throw new Error("Chapter numbers must be numbers.");
  }
  return { from, to };
}

async function handle(command, payload) {
  switch (command) {
    case "status":
      return { job: C.summarize(await loadJob()) };
    case "start": {
      if (C.isActive(await loadJob())) {
        throw new Error("A batch import is already running. Finish or cancel it first.");
      }
      const seriesId = String((payload && payload.seriesId) || "");
      if (!/^[A-Za-z0-9._-]+$/.test(seriesId)) throw new Error("A series id is required.");
      const plan = await fetchPlan({
        seriesId,
        ...readRange(payload || {}),
        includeImported: !!(payload && payload.includeImported),
      });
      const problem = C.validatePlan(plan);
      if (problem) throw new Error(problem);

      // A new job gets a new tab rather than taking over whatever the last one left.
      await chrome.storage.local.remove(TAB_KEY);
      const job = C.createJob(plan, Date.now());
      await saveJob(job);
      drive();
      return { job: C.summarize(job) };
    }
    case "resume":
    case "skip": {
      const job = await dispatch({ type: command });
      drive();
      return { job: C.summarize(job) };
    }
    case "pause":
    case "cancel":
      return { job: C.summarize(await dispatch({ type: command })) };
    case "clear": {
      // Dismiss a finished job's result. A live job is cancelled, not cleared,
      // so its place is never lost by accident.
      if (C.isActive(await loadJob())) throw new Error("Cancel the running job first.");
      await chrome.storage.local.remove([JOB_KEY, TAB_KEY]);
      publish(null);
      await syncHeartbeat(null);
      return { job: null };
    }
    default:
      throw new Error(`Unknown command: ${command}`);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.channel !== "voxJob") return undefined;
  (async () => {
    if (!(await isTrusted(sender))) throw new Error("Not allowed.");
    if (!COMMANDS.has(message.command)) throw new Error(`Unknown command: ${message.command}`);
    return handle(message.command, message.payload);
  })().then(
    (result) => sendResponse({ ok: true, ...result }),
    (error) => sendResponse({ ok: false, error: String((error && error.message) || error) })
  );
  return true; // async
});

// Live progress for the popup and the VoxNovel page.
chrome.runtime.onConnect.addListener(async (port) => {
  if (port.name !== "voxJob") return;
  if (!(await isTrusted(port.sender))) {
    port.disconnect();
    return;
  }
  ports.add(port);
  port.onDisconnect.addListener(() => ports.delete(port));
  port.postMessage({ type: "job", job: C.summarize(await loadJob()) });
});

// ------------------------------------------------------- waking events --

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === HEARTBEAT_ALARM) drive();
});

// Completing a challenge or a login usually ends with the chapter loading
// again in the job tab. That is taken as "carry on".
chrome.tabs.onUpdated.addListener(async (tabId, info, tab) => {
  if (info.status !== "complete" || tabId !== (await storedTabId())) return;
  const job = await loadJob();
  if (C.shouldAutoResume(job, tab.url)) {
    await dispatch({ type: "resume" });
    drive();
  }
});

// Closing the job tab pauses the job rather than quietly opening another.
chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (tabId !== (await storedTabId())) return;
  await chrome.storage.local.remove(TAB_KEY);
  const job = await loadJob();
  if (job && job.status === "running") await dispatch({ type: "pause" });
});

// After Chrome restarts, a job comes back paused, not running unattended.
chrome.runtime.onStartup.addListener(async () => {
  await chrome.storage.local.remove(TAB_KEY);
  const job = await loadJob();
  if (job && job.status === "running") await dispatch({ type: "pause" });
  else publish(job);
  await syncRelayRegistration();
});

// ---------------------------------------------- a custom server address --

// The relay is declared for the default localhost server. A different server
// address needs it registered at runtime, once the user has granted access.
async function syncRelayRegistration() {
  try {
    await chrome.scripting.unregisterContentScripts({ ids: [CUSTOM_RELAY_ID] });
  } catch (e) {
    // Not registered.
  }
  const origin = C.serverOrigin(await getServerUrl());
  if (!origin || C.DEFAULT_SERVER_ORIGINS.includes(origin)) return;
  if (!(await chrome.permissions.contains({ origins: [`${origin}/*`] }))) return;
  await chrome.scripting.registerContentScripts([
    { id: CUSTOM_RELAY_ID, matches: [`${origin}/*`], js: ["relay.js"], runAt: "document_start" },
  ]);
}

chrome.runtime.onInstalled.addListener(() => {
  syncRelayRegistration();
  loadJob().then(publish);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.voxServerUrl) syncRelayRegistration();
});
