function el(tag, { className, text, style } = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (style) Object.assign(node.style, style);
  return node;
}

function renderEmptyState(container, { icon, heading, headingColor, body }) {
  container.replaceChildren();
  const wrap = el("div", { className: "empty-state" });
  wrap.appendChild(el("div", { className: "empty-icon", text: icon }));
  wrap.appendChild(
    el("p", {
      text: heading,
      style: { fontWeight: "500", color: headingColor, marginBottom: "4px" },
    })
  );
  wrap.appendChild(el("p", { text: body }));
  container.appendChild(wrap);
}

document.addEventListener("DOMContentLoaded", async () => {
  const DEFAULT_SERVER_URL = "http://127.0.0.1:8000";
  const statusBadge = document.getElementById("status-badge");
  const statusText = document.getElementById("status-text");
  const cardBody = document.getElementById("card-body");
  const importBtn = document.getElementById("import-btn");
  const settingsToggle = document.getElementById("settings-toggle");
  const settingsBody = document.getElementById("settings-body");
  const serverInput = document.getElementById("server-url-input");
  const saveServerBtn = document.getElementById("save-server-btn");

  let serverUrl = DEFAULT_SERVER_URL;
  let activeDetails = null;

  // Load saved server URL
  if (chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["voxServerUrl"], (res) => {
      if (res.voxServerUrl) {
        serverUrl = res.voxServerUrl;
        serverInput.value = serverUrl;
      }
      checkServerHealth();
    });
  } else {
    checkServerHealth();
  }

  // Toggle settings
  settingsToggle.addEventListener("click", () => {
    settingsBody.classList.toggle("active");
  });

  saveServerBtn.addEventListener("click", async () => {
    const val = serverInput.value.trim().replace(/\/+$/, "");
    if (!val) return;

    let origin;
    try {
      const parsed = new URL(val);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error("Server URL must be http:// or https://");
      }
      origin = `${parsed.origin}/*`;
    } catch (e) {
      saveServerBtn.textContent = "Bad URL";
      setTimeout(() => (saveServerBtn.textContent = "Save"), 1500);
      return;
    }

    // host_permissions only covers the default localhost server, so a custom URL
    // needs an explicit grant or every fetch to it is blocked.
    try {
      const granted = await chrome.permissions.request({ origins: [origin] });
      if (!granted) {
        saveServerBtn.textContent = "Denied";
        setTimeout(() => (saveServerBtn.textContent = "Save"), 1500);
        return;
      }
    } catch (e) {
      // Already-granted origins can throw outside a user gesture; keep going.
    }

    serverUrl = val;
    chrome.storage.local.set({ voxServerUrl: serverUrl }, () => {
      saveServerBtn.textContent = "Saved!";
      setTimeout(() => (saveServerBtn.textContent = "Save"), 1500);
      checkServerHealth();
    });
  });

  async function checkServerHealth() {
    try {
      const res = await fetch(`${serverUrl}/api/extension/health`, { method: "GET" });
      if (res.ok) {
        statusBadge.className = "status-badge online";
        statusText.textContent = "Online";
      } else {
        throw new Error("HTTP " + res.status);
      }
    } catch (e) {
      statusBadge.className = "status-badge offline";
      statusText.textContent = "Offline";
    }
  }

  // Query active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) return;

  // Communicate with content script
  try {
    chrome.tabs.sendMessage(tab.id, { action: "GET_PAGE_STATUS" }, (response) => {
      if (chrome.runtime.lastError || !response || !response.ok) {
        // Tab not a content script page
        renderEmptyState(cardBody, {
          icon: "\uD83C\uDF10",
          heading: "No Novel Detected",
          headingColor: "#cbd5e1",
          body: "Open a chapter page on ReadToon (e.g. readtoon.com/content/...) to import with 1-click.",
        });
        importBtn.disabled = true;
        return;
      }

      const d = response.details;
      activeDetails = d;

      if (!d.paragraphs || d.paragraphs.length === 0) {
        renderEmptyState(cardBody, {
          icon: "\u26A0\uFE0F",
          heading: "No Content Found",
          headingColor: "#f59e0b",
          body: "Make sure this chapter is fully unlocked and readable on the page before importing.",
        });
        importBtn.disabled = true;
        return;
      }

      cardBody.replaceChildren();
      cardBody.appendChild(
        el("div", { className: "novel-title", text: d.seriesTitle || d.seriesId || "Novel" })
      );
      cardBody.appendChild(
        el("div", { className: "chapter-title", text: d.chapterTitle || "Chapter" })
      );
      const tags = el("div", { className: "meta-tags" });
      tags.appendChild(
        el("span", { className: "meta-tag", text: `\uD83D\uDCC4 ${d.paragraphs.length} Paragraphs` })
      );
      tags.appendChild(
        el("span", {
          className: "meta-tag",
          text: `\u270D\uFE0F ${Number(d.characterCount || 0).toLocaleString()} Chars`,
        })
      );
      cardBody.appendChild(tags);

      importBtn.disabled = false;
      importBtn.replaceChildren(el("span", { text: "\uD83C\uDF99\uFE0F Import Chapter to VoxNovel" }));
    });
  } catch (err) {
    console.error(err);
  }

  importBtn.addEventListener("click", () => {
    importBtn.disabled = true;
    importBtn.replaceChildren(el("span", { text: "Importing..." }));

    chrome.tabs.sendMessage(tab.id, { action: "IMPORT_NOW" }, (res) => {
      const failure = chrome.runtime.lastError
        ? chrome.runtime.lastError.message
        : !res
          ? "No response from page."
          : res.ok
            ? null
            : res.error || "Import failed.";

      if (!failure) {
        importBtn.replaceChildren(el("span", { text: "\u2705 Imported!" }));
        setTimeout(() => window.close(), 1500);
        return;
      }

      importBtn.disabled = false;
      importBtn.replaceChildren(el("span", { text: "\u274C Try Again" }));
      importBtn.title = failure;
    });
  });
});
