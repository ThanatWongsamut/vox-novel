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

  saveServerBtn.addEventListener("click", () => {
    const val = serverInput.value.trim().replace(/\/+$/, "");
    if (val) {
      serverUrl = val;
      chrome.storage.local.set({ voxServerUrl: serverUrl }, () => {
        saveServerBtn.textContent = "Saved!";
        setTimeout(() => (saveServerBtn.textContent = "Save"), 1500);
        checkServerHealth();
      });
    }
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
        cardBody.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">🌐</div>
            <p style="font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">No Novel Detected</p>
            <p>Open a chapter page on ReadToon (e.g. readtoon.com/content/...) to import with 1-click.</p>
          </div>
        `;
        importBtn.disabled = true;
        return;
      }

      const d = response.details;
      activeDetails = d;

      if (!d.paragraphs || d.paragraphs.length === 0) {
        cardBody.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <p style="font-weight: 500; color: #f59e0b; margin-bottom: 4px;">No Content Found</p>
            <p>Make sure this chapter is fully unlocked and readable on the page before importing.</p>
          </div>
        `;
        importBtn.disabled = true;
        return;
      }

      cardBody.innerHTML = `
        <div class="novel-title">${d.seriesTitle || d.seriesId || "Novel"}</div>
        <div class="chapter-title">${d.chapterTitle || "Chapter"}</div>
        <div class="meta-tags">
          <span class="meta-tag">📄 ${d.paragraphs.length} Paragraphs</span>
          <span class="meta-tag">✍️ ${d.characterCount.toLocaleString()} Chars</span>
        </div>
      `;

      importBtn.disabled = false;
      importBtn.innerHTML = `<span>🎙️ Import Chapter to VoxNovel</span>`;
    });
  } catch (err) {
    console.error(err);
  }

  importBtn.addEventListener("click", () => {
    importBtn.disabled = true;
    importBtn.innerHTML = `<span>Importing...</span>`;

    chrome.tabs.sendMessage(tab.id, { action: "IMPORT_NOW" }, (res) => {
      if (res && res.ok) {
        importBtn.innerHTML = `<span>✅ Imported!</span>`;
        setTimeout(() => window.close(), 1500);
      } else {
        importBtn.disabled = false;
        importBtn.innerHTML = `<span>❌ Try Again</span>`;
      }
    });
  });
});
