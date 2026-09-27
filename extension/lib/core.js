// VoxNovel extension core: pure logic shared by the background worker, the
// ReadToon content script and the Node test suite. No chrome.* calls here --
// everything that touches the browser lives in background.js and content.js.

(function (root) {
  "use strict";

  const DEFAULT_SERVER_URL = "http://127.0.0.1:8000";
  const DEFAULT_SERVER_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"];

  // Delay between chapters. Politeness to the site, not disguise: the driver
  // does not randomise it or otherwise pretend to be a person.
  const CHAPTER_DELAY_MS = 3000;
  // How long a chapter page gets to render its text before we decide.
  const EXTRACT_TIMEOUT_MS = 15000;
  // An empty page is reloaded this many times before it is skipped.
  const MAX_ATTEMPTS = 2;
  const MAX_JOB_CHAPTERS = 5000;

  // Shown by ReadToon instead of the text when a chapter is not owned.
  const PURCHASE_MARKERS = ["ยืนยันการซื้อตอน", "เหรียญไม่เพียงพอ"];
  // ...and when the reader is not logged in.
  const LOGIN_MARKERS = ["เข้าสู่ระบบเพื่อซื้อ"];

  /**
   * True only for a ReadToon chapter page: https, readtoon.com or a subdomain,
   * /content/<slug>/<number>. A batch job may open nothing else, so a web page
   * that reaches the extension cannot point it at an arbitrary site.
   */
  function isReadtoonChapterUrl(candidate) {
    let url;
    try {
      url = new URL(candidate);
    } catch (e) {
      return false;
    }
    if (url.protocol !== "https:") return false;
    if (url.username || url.password || url.port) return false;
    const host = url.hostname.toLowerCase();
    if (host !== "readtoon.com" && !host.endsWith(".readtoon.com")) return false;
    return /^\/content\/[^/]+\/\d+(?:\.\d+)?\/?$/.test(url.pathname);
  }

  /**
   * The series a ReadToon series or chapter page belongs to, or null.
   * `/content/<slug>` and `/content/<slug>/<n>` both give `<slug>`.
   */
  function readtoonSeriesId(candidate) {
    let url;
    try {
      url = new URL(candidate);
    } catch (e) {
      return null;
    }
    const host = url.hostname.toLowerCase();
    if (url.protocol !== "https:") return null;
    if (host !== "readtoon.com" && !host.endsWith(".readtoon.com")) return null;
    const match = url.pathname.match(/^\/content\/([A-Za-z0-9._-]+)(?:\/[^/]*)?\/?$/);
    return match ? match[1] : null;
  }

  /** The origin part of a server URL, or null if it is not http(s). */
  function serverOrigin(candidate) {
    try {
      const url = new URL(candidate);
      if (url.protocol !== "http:" && url.protocol !== "https:") return null;
      return url.origin;
    } catch (e) {
      return null;
    }
  }

  /**
   * Decide what a chapter page is showing.
   *
   *   ok        the chapter text is there
   *   locked    ReadToon wants the chapter bought
   *   login     ReadToon wants the reader logged in
   *   challenge a Turnstile or other challenge is up
   *   pending   nothing conclusive yet -- keep waiting
   *   empty     timed out with nothing recognisable
   *
   * Text wins over everything: a page that shows the chapter is fine whatever
   * else is on it, including an invisible Turnstile widget. A purchase or login
   * prompt does not go away by waiting, so it is reported at once. A challenge
   * may clear itself, so it is only reported once the wait has run out.
   */
  function classifyPage(signals) {
    const s = signals || {};
    if ((s.paragraphCount || 0) > 0 && (s.characterCount || 0) > 0) return "ok";
    const text = s.bodyText || "";
    if (LOGIN_MARKERS.some((m) => text.includes(m))) return "login";
    if (PURCHASE_MARKERS.some((m) => text.includes(m))) return "locked";
    if (!s.timedOut) return "pending";
    return s.challengeVisible ? "challenge" : "empty";
  }

  /** The body the server's /api/extension/import expects. */
  function buildImportPayload(details) {
    return {
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
  }

  /**
   * Check a plan returned by the server before a single page is opened.
   * Returns an error message, or null when the whole plan is acceptable --
   * one bad URL rejects everything rather than being quietly dropped.
   */
  function validatePlan(plan) {
    if (!plan || !Array.isArray(plan.chapters)) return "The server returned no chapter list.";
    if (plan.chapters.length > MAX_JOB_CHAPTERS) {
      return `Too many chapters in one job (${plan.chapters.length}).`;
    }
    for (const ch of plan.chapters) {
      if (!ch || typeof ch.no !== "number" || !Number.isFinite(ch.no)) {
        return "A chapter in the plan has no number.";
      }
      if (!isReadtoonChapterUrl(ch.url)) {
        return `Refusing a chapter URL that is not a ReadToon chapter page: ${ch.url}`;
      }
    }
    return null;
  }

  function createJob(plan, now) {
    return {
      id: `job-${now}`,
      seriesId: plan.series_id,
      seriesTitle: plan.series_title || plan.series_id,
      status: plan.chapters.length ? "running" : "done",
      queue: plan.chapters.map((ch) => ({
        no: ch.no,
        url: ch.url,
        title: ch.title || "",
        isLocked: !!ch.is_locked,
      })),
      index: 0,
      attempts: 0,
      imported: [],
      skipped: [],
      pause: null,
      alreadyImported: plan.already_imported || 0,
      startedAt: now,
      updatedAt: now,
    };
  }

  function current(job) {
    return job && job.index < job.queue.length ? job.queue[job.index] : null;
  }

  function advance(job) {
    const index = job.index + 1;
    return {
      ...job,
      index,
      attempts: 0,
      pause: null,
      status: index >= job.queue.length ? "done" : "running",
    };
  }

  // Outcomes only the user can resolve. Anything else is retried, then skipped.
  const PAUSING_OUTCOMES = ["locked", "login", "challenge", "server", "redirected"];

  const PAUSE_MESSAGES = {
    locked: "This chapter is not unlocked on your account. Buy it on ReadToon and press Continue, or Skip it.",
    login: "ReadToon wants you logged in. Log in on this tab and press Continue, or Skip this chapter.",
    challenge: "ReadToon is showing a check. Complete it on this tab; the import continues once the chapter loads.",
    server: "Could not reach the VoxNovel server. Start it (uv run vox-novel web) and press Continue.",
    redirected: "ReadToon sent this tab to a different page instead of the chapter. Sort it out on this tab and press Continue, or Skip this chapter.",
    user: "Paused.",
  };

  /**
   * The job as a pure state machine. Every change goes through here, so the
   * background worker can be suspended and restarted between any two events
   * without losing its place.
   */
  function reduce(job, event, now) {
    if (!job) return job;
    const stamp = (next) => ({ ...next, updatedAt: now });
    const chapter = current(job);

    switch (event.type) {
      case "outcome": {
        if (job.status !== "running" || !chapter) return job;
        const outcome = event.outcome;
        if (outcome === "imported") {
          return stamp(advance({ ...job, imported: [...job.imported, chapter.no] }));
        }
        if (PAUSING_OUTCOMES.includes(outcome)) {
          return stamp({
            ...job,
            status: "paused",
            pause: { reason: outcome, no: chapter.no, url: chapter.url, message: PAUSE_MESSAGES[outcome] },
          });
        }
        // Nothing recognisable: reload once more, then give up on this one.
        const attempts = job.attempts + 1;
        if (attempts < MAX_ATTEMPTS) return stamp({ ...job, attempts });
        return stamp(
          advance({
            ...job,
            skipped: [...job.skipped, { no: chapter.no, reason: event.detail || "no chapter text found" }],
          })
        );
      }
      case "pause":
        if (job.status !== "running") return job;
        return stamp({
          ...job,
          status: "paused",
          pause: { reason: "user", no: chapter && chapter.no, url: chapter && chapter.url, message: PAUSE_MESSAGES.user },
        });
      case "resume":
        if (job.status !== "paused") return job;
        return stamp({ ...job, status: "running", pause: null, attempts: 0 });
      case "skip": {
        if (job.status !== "paused" || !chapter) return job;
        const reason = job.pause ? job.pause.reason : "skipped";
        return stamp(advance({ ...job, skipped: [...job.skipped, { no: chapter.no, reason }] }));
      }
      case "cancel":
        if (job.status === "done" || job.status === "cancelled") return job;
        return stamp({ ...job, status: "cancelled", pause: null });
      default:
        return job;
    }
  }

  /** True while a job still needs the driver: running or waiting on the user. */
  function isActive(job) {
    return !!job && (job.status === "running" || job.status === "paused");
  }

  /**
   * Whether a page load in the job tab should resume a paused job by itself.
   * Only for a challenge or login, and only on the chapter being waited for --
   * that page load is how completing one usually shows. Buying a chapter never
   * resumes on its own: that is a deliberate decision, confirmed by Continue.
   */
  function shouldAutoResume(job, loadedUrl) {
    if (!job || job.status !== "paused" || !job.pause) return false;
    if (job.pause.reason !== "challenge" && job.pause.reason !== "login") return false;
    return samePage(loadedUrl, job.pause.url);
  }

  function samePage(a, b) {
    try {
      const x = new URL(a);
      const y = new URL(b);
      return x.origin === y.origin && x.pathname.replace(/\/$/, "") === y.pathname.replace(/\/$/, "");
    } catch (e) {
      return false;
    }
  }

  /** A short summary for the popup and the VoxNovel page. */
  function summarize(job) {
    if (!job) return null;
    const ch = current(job);
    return {
      id: job.id,
      seriesId: job.seriesId,
      seriesTitle: job.seriesTitle,
      status: job.status,
      total: job.queue.length,
      position: Math.min(job.index + 1, job.queue.length),
      current: ch ? { no: ch.no, title: ch.title, isLocked: ch.isLocked } : null,
      imported: job.imported.length,
      skipped: job.skipped,
      alreadyImported: job.alreadyImported,
      pause: job.pause,
    };
  }

  const api = {
    DEFAULT_SERVER_URL,
    DEFAULT_SERVER_ORIGINS,
    CHAPTER_DELAY_MS,
    EXTRACT_TIMEOUT_MS,
    MAX_ATTEMPTS,
    MAX_JOB_CHAPTERS,
    PURCHASE_MARKERS,
    LOGIN_MARKERS,
    isReadtoonChapterUrl,
    readtoonSeriesId,
    serverOrigin,
    classifyPage,
    buildImportPayload,
    validatePlan,
    createJob,
    current,
    reduce,
    isActive,
    shouldAutoResume,
    samePage,
    summarize,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.VoxCore = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
