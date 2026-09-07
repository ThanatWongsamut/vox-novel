// VoxNovel Content Script - 1-Click Novel Importer

(function () {
  "use strict";

  const DEFAULT_SERVER_URL = "http://127.0.0.1:8000";

  async function getServerUrl() {
    return new Promise((resolve) => {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.get(["voxServerUrl"], (result) => {
          resolve(result.voxServerUrl || DEFAULT_SERVER_URL);
        });
      } else {
        resolve(DEFAULT_SERVER_URL);
      }
    });
  }

  function extractPageDetails() {
    const url = window.location.href;
    const pathParts = window.location.pathname.split("/").filter(Boolean);

    let seriesId = "";
    let chapterNo = null;
    let seriesTitle = "";
    let chapterTitle = "";

    // ReadToon URL format: /content/{series-slug}/{chapter-number}
    if (pathParts.length >= 2 && pathParts[0] === "content") {
      seriesId = pathParts[1];
      if (pathParts[2]) {
        const num = parseFloat(pathParts[2]);
        if (!isNaN(num)) chapterNo = num;
      }
    }

    // 1. Extract Series Title (h1 or breadcrumb or docTitle)
    const h1 = document.querySelector("h1");
    if (h1 && h1.textContent.trim()) {
      seriesTitle = h1.textContent.trim();
    }
    const escapedSeriesId = seriesId ? CSS.escape(seriesId) : "";
    const breadcrumbSeriesLink = escapedSeriesId
      ? document.querySelector(`a[href="/content/${escapedSeriesId}"]`) ||
        document.querySelector(`a[href*="/content/${escapedSeriesId}"]`)
      : null;
    if (breadcrumbSeriesLink && breadcrumbSeriesLink.textContent.trim()) {
      seriesTitle = breadcrumbSeriesLink.textContent.trim();
    }

    // 2. Extract Chapter Title (from page DOM sub-heading or document.title)
    // On ReadToon, the chapter title is in p.text-default-500: e.g. "ตอนที่ 170 - ฉันถูกเข้าใจผิดว่าเป็นผี0170"
    const domChapEl = Array.from(document.querySelectorAll("p, span, h2, h3, div")).find((el) => {
      const t = el.textContent.trim();
      if (t.length > 100) return false;
      if (chapterNo !== null && (t.startsWith(`ตอนที่ ${chapterNo}`) || t.startsWith(`ตอนที่${chapterNo}`))) {
        return true;
      }
      return false;
    });
    if (domChapEl) {
      chapterTitle = domChapEl.textContent.trim();
    }

    // Fallback: match from document.title
    // Format on ReadToon: "Series Title - ตอนที่ 170 - Subtitle - ReadToon"
    const docTitle = document.title || "";
    if (!chapterTitle && docTitle) {
      const m = docTitle.match(/(?:^|\s*-\s*)(ตอนที่\s*\d+.*?)(?:\s*-\s*ReadToon|$)/i);
      if (m) {
        chapterTitle = m[1].trim();
      }
    }

    // Ensure chapterTitle is not mistakenly the whole novel title
    if (!chapterTitle || chapterTitle === seriesTitle) {
      chapterTitle = chapterNo !== null ? `ตอนที่ ${chapterNo}` : "Chapter";
    }

    // Extract novel cover from meta or page
    let coverUrl = null;
    const ogImage = document.querySelector('meta[property="og:image"]');
    if (ogImage && ogImage.content && !ogImage.content.includes("logo") && !ogImage.content.includes("banner")) {
      coverUrl = ogImage.content;
    }

    // Extract prose content (matching readtoon.py scraper logic)
    const proseEl = document.querySelector("div.prose.mx-auto") ||
                    document.querySelector("div.prose") ||
                    document.querySelector("article") ||
                    document.querySelector(".chapter-content");

    let paragraphs = [];
    if (proseEl) {
      // Clone element to cleanly convert <br> and <p> into consistent paragraphs
      const clone = proseEl.cloneNode(true);
      clone.querySelectorAll("br").forEach((br) => br.replaceWith("\n"));
      clone.querySelectorAll("p").forEach((p) => p.insertAdjacentText("afterend", "\n\n"));
      const rawText = clone.textContent || clone.innerText || "";
      paragraphs = rawText
        .split(/\n+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
    }

    return {
      url,
      seriesId,
      seriesTitle,
      chapterNo,
      chapterTitle,
      coverUrl,
      paragraphs,
      characterCount: paragraphs.reduce((sum, p) => sum + p.length, 0),
    };
  }

  function isSafeHttpUrl(candidate) {
    try {
      const parsed = new URL(candidate);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch (e) {
      return false;
    }
  }

  function showToast({ type = "info", title, message, actionText, actionUrl }) {
    const existing = document.getElementById("vox-novel-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.id = "vox-novel-toast";
    toast.className = type;

    const header = document.createElement("div");
    header.className = `vox-toast-header ${type}`;

    const titleEl = document.createElement("span");
    titleEl.textContent = title || "";
    header.appendChild(titleEl);

    const closeEl = document.createElement("span");
    closeEl.textContent = "\u2715";
    closeEl.style.cursor = "pointer";
    closeEl.style.opacity = "0.7";
    closeEl.addEventListener("click", () => toast.remove());
    header.appendChild(closeEl);

    const body = document.createElement("div");
    body.className = "vox-toast-body";
    body.textContent = message || "";

    toast.appendChild(header);
    toast.appendChild(body);

    // Only same-origin http(s) links from our own server are rendered as actions.
    if (actionText && actionUrl && isSafeHttpUrl(actionUrl)) {
      const link = document.createElement("a");
      link.href = actionUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "vox-toast-action";
      link.textContent = actionText;
      toast.appendChild(link);
    }

    document.body.appendChild(toast);

    if (type !== "error") {
      setTimeout(() => {
        if (toast.parentNode) {
          toast.style.opacity = "0";
          toast.style.transform = "translateY(-16px)";
          setTimeout(() => toast.remove(), 300);
        }
      }, 7000);
    }
  }

  async function performImport() {
    const btn = document.getElementById("vox-novel-import-btn");
    if (!btn) return { ok: false, error: "Import button not available on this page." };

    const originalHtml = btn.innerHTML;
    btn.classList.add("loading");
    btn.innerHTML = `<span class="vox-spinner"></span><span>Importing Chapter...</span>`;

    const details = extractPageDetails();

    if (!details.paragraphs || details.paragraphs.length === 0) {
      btn.classList.remove("loading");
      btn.innerHTML = originalHtml;
      showToast({
        type: "error",
        title: "Content Not Found",
        message: "Could not find chapter text in page. If this is a paid chapter, make sure it is unlocked and visible.",
      });
      return { ok: false, error: "No chapter content found on page." };
    }

    const serverUrl = await getServerUrl();

    try {
      const payload = {
        url: details.url,
        series_id: details.seriesId,
        series_title: details.seriesTitle,
        chapter_no: details.chapterNo,
        chapter_title: details.chapterTitle,
        paragraphs: details.paragraphs,
        cover_url: details.coverUrl,
        source: "readtoon",
        source_language: "th",
      };

      const res = await fetch(`${serverUrl}/api/extension/import`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
      }

      const data = await res.json();

      btn.classList.remove("loading");
      btn.innerHTML = `<span>✅ Chapter Imported</span>`;
      btn.style.background = "linear-gradient(135deg, #10b981 0%, #059669 100%)";

      const readerFullUrl = `${serverUrl}${data.read_url}`;

      showToast({
        type: "success",
        title: "Import Complete!",
        message: `Successfully saved ${details.chapterTitle || "Chapter"} (${details.paragraphs.length} paragraphs).`,
        actionText: "Open in VoxNovel Reader \u2197",
        actionUrl: readerFullUrl,
      });

      setTimeout(() => {
        btn.innerHTML = originalHtml;
        btn.style.background = "";
      }, 5000);

      return { ok: true, readUrl: readerFullUrl };
    } catch (err) {
      console.error("[VoxNovel]", err);
      btn.classList.remove("loading");
      btn.innerHTML = `<span>❌ Import Failed</span>`;
      btn.style.background = "linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)";

      showToast({
        type: "error",
        title: "Connection Error",
        message: `${err.message}. Is VoxNovel web server running at ${serverUrl}? (uv run vox-novel web)`,
      });

      setTimeout(() => {
        btn.innerHTML = originalHtml;
        btn.style.background = "";
      }, 4000);

      return { ok: false, error: err.message };
    }
  }

  function syncFloatingButton() {
    const existing = document.getElementById("vox-novel-import-btn");
    const onChapterPage = window.location.pathname.includes("/content/");

    // Client-side navigation can take us off a chapter page: drop a stale button.
    if (!onChapterPage) {
      if (existing) existing.remove();
      return;
    }
    if (existing) return;

    const btn = document.createElement("button");
    btn.id = "vox-novel-import-btn";
    btn.title = "Import this chapter directly into VoxNovel for reading and VoxCPM2 TTS";
    btn.innerHTML = `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="22"></line>
      </svg>
      <span>Import to VoxNovel</span>
    `;

    btn.addEventListener("click", (e) => {
      e.preventDefault();
      performImport();
    });

    document.body.appendChild(btn);
  }

  // Handle messages from popup
  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "GET_PAGE_STATUS") {
        const details = extractPageDetails();
        sendResponse({ ok: true, details });
      } else if (request.action === "IMPORT_NOW") {
        performImport().then(
          (result) => sendResponse(result || { ok: false, error: "Import produced no result." }),
          (err) => sendResponse({ ok: false, error: String((err && err.message) || err) })
        );
        return true; // async
      }
    });
  }

  // Inject when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", syncFloatingButton);
  } else {
    syncFloatingButton();
  }

  // Next.js navigates client-side. A content script runs in an isolated world, so
  // it cannot patch the page's history.pushState -- poll the URL instead, and stop
  // once the page goes away.
  let lastUrl = window.location.href;
  function onLocationChange() {
    if (window.location.href === lastUrl) return;
    lastUrl = window.location.href;
    syncFloatingButton();
  }

  const urlPoll = setInterval(onLocationChange, 1000);
  window.addEventListener("popstate", onLocationChange);
  window.addEventListener("hashchange", onLocationChange);
  window.addEventListener("pagehide", () => clearInterval(urlPoll), { once: true });
})();
