// VoxNovel relay: runs on the VoxNovel web UI and lets that page reach the
// extension, so "Import missing chapters" can be started from the series page.
//
// It only forwards a command name and its arguments. What a command may do is
// decided in background.js, which fetches the chapter list from the server
// itself -- so this page can ask for a job but cannot choose which URLs are
// opened.

(function () {
  "use strict";

  if (window.__voxNovelRelay) return;
  window.__voxNovelRelay = true;

  const FROM_PAGE = "voxnovel-page";
  const FROM_EXTENSION = "voxnovel-extension";
  const version = chrome.runtime.getManifest().version;

  function toPage(message) {
    window.postMessage({ source: FROM_EXTENSION, ...message }, window.location.origin);
  }

  // Lets the page show the button only when the extension is installed.
  document.documentElement.dataset.voxnovelExtension = version;

  let port = null;
  let watching = false;

  const finished = (job) => !job || job.status === "done" || job.status === "cancelled";

  // Progress is streamed only while there is a job to follow. Each connection
  // wakes the extension's background worker, so an open VoxNovel tab with
  // nothing running must not keep reconnecting.
  function watch() {
    watching = true;
    if (port) return;
    try {
      port = chrome.runtime.connect({ name: "voxJob" });
    } catch (e) {
      // The extension was reloaded or removed; this page is orphaned.
      watching = false;
      toPage({ type: "gone" });
      return;
    }
    port.onMessage.addListener((message) => {
      if (!message || message.type !== "job") return;
      toPage({ type: "job", job: message.job });
      if (finished(message.job)) {
        watching = false;
        port.disconnect();
        port = null;
      }
    });
    port.onDisconnect.addListener(() => {
      port = null;
      // The worker was suspended mid-job. Reconnecting wakes it again.
      if (watching) setTimeout(watch, 1000);
    });
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data;
    if (!data || data.source !== FROM_PAGE || typeof data.command !== "string") return;

    if (data.command === "watch") {
      watch();
      return;
    }

    const reply = (body) => toPage({ type: "reply", requestId: data.requestId, ...body });
    try {
      chrome.runtime
        .sendMessage({ channel: "voxJob", command: data.command, payload: data.payload || {} })
        .then(
          (response) => reply(response || { ok: false, error: "No response from the extension." }),
          (error) => reply({ ok: false, error: String((error && error.message) || error) })
        );
    } catch (e) {
      reply({ ok: false, error: "The VoxNovel extension was reloaded. Refresh this page." });
    }
  });

  toPage({ type: "ready", version });
})();
