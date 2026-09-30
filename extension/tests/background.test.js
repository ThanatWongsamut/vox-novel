// Runs background.js against a stand-in for the chrome.* APIs, so whole jobs
// can be driven -- start, pause, resume, skip -- without a browser.
// Run with: node --test extension/tests/
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const EXT = path.join(__dirname, "..");
const SERVER = "http://127.0.0.1:8000";
const RUNTIME_ID = "vox-test-extension";

const chapterUrl = (no) => `https://readtoon.com/content/s/${no}`;

function event() {
  const listeners = [];
  return {
    listeners,
    addListener: (f) => listeners.push(f),
    removeListener: (f) => {
      const i = listeners.indexOf(f);
      if (i >= 0) listeners.splice(i, 1);
    },
    fire: (...args) => listeners.map((f) => f(...args)),
  };
}

/**
 * A fake browser. `pages` maps a chapter URL to what content.js would report
 * there: "ok", "locked", "challenge", ... or { redirectTo } for a tab that
 * lands somewhere else.
 */
function makeBrowser({ plan, pages = {}, importStatus = () => 200,
  beforeRead = async () => {}, beforePlan = async () => {}, beforeImport = async () => {},
  beforeFocus = async () => {},
} = {}) {
  const store = {};
  const tabs = new Map();
  let nextTab = 1;
  const imports = [];
  const closed = [];
  const planQueries = [];
  const opened = [];
  const focused = [];

  const onUpdated = event();
  const onRemoved = event();
  const onMessage = event();

  function load(tab, url) {
    const page = pages[url] || "ok";
    tab.url = page && page.redirectTo ? page.redirectTo : url;
    opened.push(url);
    // Pages finish loading asynchronously, as they do in Chrome.
    setImmediate(() => onUpdated.fire(tab.id, { status: "complete" }, { ...tab }));
  }

  const chrome = {
    runtime: {
      id: RUNTIME_ID,
      onMessage,
      onConnect: event(),
      onStartup: event(),
      onInstalled: event(),
      getManifest: () => ({ version: "test" }),
    },
    storage: {
      local: {
        get: async (keys) => {
          const list = typeof keys === "string" ? [keys] : keys;
          return Object.fromEntries(list.filter((k) => k in store).map((k) => [k, structuredClone(store[k])]));
        },
        set: async (items) => Object.assign(store, structuredClone(items)),
        // Like Chrome: one key or a list of them.
        remove: async (keys) => {
          for (const key of [].concat(keys)) delete store[key];
        },
      },
      onChanged: event(),
    },
    tabs: {
      onUpdated,
      onRemoved,
      create: async ({ url }) => {
        const tab = { id: nextTab++, windowId: 1, url: "about:blank" };
        tabs.set(tab.id, tab);
        load(tab, url);
        return { ...tab };
      },
      update: async (id, props) => {
        const tab = tabs.get(id);
        if (!tab) throw new Error("No tab with id");
        if (props.url) load(tab, props.url);
        if (props.active) focused.push(tab.url);
        return { ...tab };
      },
      remove: async (id) => {
        if (!tabs.delete(id)) throw new Error("No tab with id");
        closed.push(id);
        onRemoved.fire(id, {});
      },
      get: async (id) => {
        const tab = tabs.get(id);
        if (!tab) throw new Error("No tab with id");
        return { ...tab };
      },
      sendMessage: async (id, message) => {
        const tab = tabs.get(id);
        assert.equal(message.action, "EXTRACT_WHEN_READY");
        await beforeRead(tab.url);
        const page = pages[tab.url] || "ok";
        if (page === "ok") {
          return {
            ok: true,
            status: "ok",
            details: {
              url: tab.url, seriesId: "s", seriesTitle: "Series",
              chapterNo: Number(tab.url.split("/").pop()), chapterTitle: "t",
              paragraphs: ["text"], coverUrl: null, characterCount: 4,
            },
          };
        }
        return { ok: true, status: page, details: { url: tab.url } };
      },
    },
    windows: { update: async () => { await beforeFocus(); return {}; } },
    alarms: { create: async () => {}, clear: async () => {}, onAlarm: event() },
    action: { setBadgeText: () => {}, setBadgeBackgroundColor: () => {} },
    scripting: { registerContentScripts: async () => {}, unregisterContentScripts: async () => {} },
    permissions: { contains: async () => false },
  };

  async function fetch(url, init = {}) {
    const u = new URL(url);
    if (u.pathname === "/api/extension/plan") {
      planQueries.push(Object.fromEntries(u.searchParams));
      await beforePlan(u);
      return { ok: true, status: 200, json: async () => structuredClone(plan) };
    }
    if (u.pathname === "/api/extension/import") {
      const body = JSON.parse(init.body);
      await beforeImport(body, init);
      const status = importStatus(body);
      if (status === 200) imports.push(body.chapter_no);
      return { ok: status === 200, status, json: async () => ({ detail: `HTTP ${status}` }) };
    }
    throw new Error(`unexpected fetch ${url}`);
  }

  const context = vm.createContext({
    chrome, fetch, URL, URLSearchParams, console, structuredClone,
    setTimeout, clearTimeout, setImmediate, AbortController,
    crypto: require("node:crypto").webcrypto,
  });
  context.self = context;
  context.globalThis = context;
  context.importScripts = (file) => {
    vm.runInContext(fs.readFileSync(path.join(EXT, file), "utf8"), context, { filename: file });
  };
  vm.runInContext(fs.readFileSync(path.join(EXT, "background.js"), "utf8"), context, {
    filename: "background.js",
  });
  // Tests do not wait three seconds per chapter.
  context.VoxCore.CHAPTER_DELAY_MS = 0;

  async function send(command, payload, sender = { id: RUNTIME_ID }) {
    return new Promise((resolve) => {
      const handled = onMessage.listeners.map((f) =>
        f({ channel: "voxJob", command, payload }, sender, resolve)
      );
      if (!handled.includes(true)) resolve(undefined);
    });
  }

  async function job() {
    return (await send("status")).job;
  }

  /** Wait until the job reaches a status, or fail. */
  async function until(status, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const j = await job();
      if (j && j.status === status) return j;
      if (Date.now() > deadline) assert.fail(`job never became ${status}; it is ${j && j.status}`);
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  function reload(url) {
    // What completing a challenge usually looks like: the chapter loads again.
    const tab = [...tabs.values()][0];
    load(tab, url);
  }

  function openTabs() {
    return [...tabs.values()];
  }

  return { send, job, until, imports, opened, focused, pages, reload, store, planQueries, closed, openTabs };
}

function planFor(...nos) {
  return {
    series_id: "s",
    series_title: "Series",
    already_imported: 0,
    chapters: nos.map((no) => ({ no, url: chapterUrl(no), title: `ตอนที่ ${no}`, is_locked: false })),
  };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test("overlapping starts refuse the second request while a plan is pending", async () => {
  const entered = deferred(), release = deferred();
  let plans = 0;
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" },
    beforePlan: async () => {
      if (++plans === 1) { entered.resolve(); await release.promise; }
    },
  });
  const first = b.send("start", { seriesId: "s" });
  await entered.promise;
  const second = await b.send("start", { seriesId: "s" });
  release.resolve();
  assert.equal((await first).ok, true);
  assert.equal(second.ok, false);
  assert.equal(b.planQueries.length, 1);
  await b.until("paused");
});

test("a failed plan releases the start reservation", async () => {
  let fail = true;
  const b = makeBrowser({ plan: planFor(1), beforePlan: async () => {
    if (fail) throw new Error("offline");
  } });
  assert.equal((await b.send("start", { seriesId: "s" })).ok, false);
  fail = false;
  assert.equal((await b.send("start", { seriesId: "s" })).ok, true);
  await b.until("done");
});

test("cancel during extraction prevents the import request", async () => {
  const entered = deferred(), release = deferred(), returned = deferred();
  const b = makeBrowser({ plan: planFor(1), beforeRead: async () => {
    entered.resolve(); await release.promise; returned.resolve();
  } });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  await b.send("cancel");
  release.resolve();
  await returned.promise;
  // Flush the driver's promise continuations after the content reply.
  await new Promise(setImmediate);
  assert.equal((await b.job()).status, "cancelled");
  assert.deepEqual(b.imports, []);
});

test("pause then skip discards the old extraction and imports the next chapter", async () => {
  const entered = deferred(), release = deferred();
  const b = makeBrowser({ plan: planFor(1, 2), beforeRead: async (url) => {
    if (url === chapterUrl(1)) { entered.resolve(); await release.promise; }
  } });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  await b.send("pause");
  await b.send("skip");
  release.resolve();
  const done = await b.until("done");
  assert.deepEqual(b.imports, [2]);
  assert.equal(done.skipped[0].no, 1);
});

test("pause then resume invalidates the first extraction even at the same chapter", async () => {
  const entered = deferred(), release = deferred();
  let reads = 0;
  const b = makeBrowser({ plan: planFor(1), beforeRead: async () => {
    reads += 1;
    if (reads === 1) { entered.resolve(); await release.promise; }
  } });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  await b.send("pause");
  await b.send("resume");
  release.resolve();
  await b.until("done");
  assert.equal(reads, 2);
  assert.deepEqual(b.imports, [1]);
});

test("cancel aborts an import request already waiting on the server", { timeout: 1000 }, async () => {
  const entered = deferred(), aborted = deferred();
  const b = makeBrowser({ plan: planFor(1), beforeImport: async (body, init) => {
    entered.resolve();
    await new Promise((resolve, reject) => init.signal.addEventListener("abort", () => {
      aborted.resolve(); reject(new Error("aborted"));
    }, { once: true }));
  } });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  await b.send("cancel");
  await aborted.promise;
  await new Promise(setImmediate);
  assert.equal((await b.job()).status, "cancelled");
  assert.deepEqual(b.imports, []);
});

test("overlapping cancel and pause cannot resurrect a cancelled job", async () => {
  const entered = deferred(), release = deferred();
  const b = makeBrowser({ plan: planFor(1), beforeRead: async () => {
    entered.resolve(); await release.promise;
  } });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  await Promise.all([b.send("cancel"), b.send("pause")]);
  release.resolve();
  await new Promise(setImmediate);
  assert.equal((await b.job()).status, "cancelled");
  assert.deepEqual(b.imports, []);
});

test("a new job cannot receive the cancelled job's pending extraction", async () => {
  const entered = deferred(), release = deferred();
  let reads = 0;
  const b = makeBrowser({ plan: planFor(1), beforeRead: async () => {
    if (++reads === 1) { entered.resolve(); await release.promise; }
  } });
  const first = await b.send("start", { seriesId: "s" });
  await entered.promise;
  await b.send("cancel");
  const second = await b.send("start", { seriesId: "s" });
  assert.notEqual(second.job.id, first.job.id);
  release.resolve();
  await b.until("done");
  assert.equal(reads, 2);
  assert.deepEqual(b.imports, [1]);
});

test("resume while the previous driver is finishing its pause continues immediately", async () => {
  const entered = deferred(), release = deferred();
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" },
    beforeFocus: async () => { entered.resolve(); await release.promise; },
  });
  await b.send("start", { seriesId: "s" });
  await entered.promise;
  b.pages[chapterUrl(1)] = "ok";
  await b.send("resume");
  release.resolve();
  await b.until("done");
  assert.deepEqual(b.imports, [1]);
});

test("imports every chapter, in order, in one tab", async () => {
  const b = makeBrowser({ plan: planFor(1, 2, 3) });
  const reply = await b.send("start", { seriesId: "s" });
  assert.equal(reply.ok, true, reply.error);
  const job = await b.until("done");
  assert.deepEqual(b.imports, [1, 2, 3]);
  assert.equal(job.imported, 3);
});

test("a single chapter is a job of one", async () => {
  const b = makeBrowser({ plan: planFor(7) });
  await b.send("start", { seriesId: "s", from: 7, to: 7 });
  await b.until("done");
  assert.deepEqual(b.imports, [7]);
});

test("a re-fetch asks for chapters already imported; a normal job does not", async () => {
  const b = makeBrowser({ plan: planFor(5) });
  await b.send("start", { seriesId: "s", from: 5, to: 5, includeImported: true });
  await b.until("done");
  assert.equal(b.planQueries[0].include_imported, "true");
  assert.deepEqual(b.imports, [5]);

  const normal = makeBrowser({ plan: planFor(5) });
  await normal.send("start", { seriesId: "s", from: 5, to: 5 });
  await normal.until("done");
  assert.equal(normal.planQueries[0].include_imported, undefined);
});

test("an unowned paid chapter pauses, brings the tab forward, and can be skipped", async () => {
  const b = makeBrowser({ plan: planFor(1, 2, 3), pages: { [chapterUrl(2)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  const paused = await b.until("paused");
  assert.equal(paused.pause.reason, "locked");
  assert.equal(paused.current.no, 2);
  assert.deepEqual(b.imports, [1]);
  assert.ok(b.focused.includes(chapterUrl(2)), "the tab needing the user was not brought forward");

  await b.send("skip");
  const done = await b.until("done");
  assert.deepEqual(b.imports, [1, 3]);
  assert.deepEqual(done.skipped, [{ no: 2, reason: "locked" }]);
});

test("after buying, Continue imports the same chapter", async () => {
  const b = makeBrowser({ plan: planFor(1, 2), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  b.pages[chapterUrl(1)] = "ok"; // bought
  await b.send("resume");
  await b.until("done");
  assert.deepEqual(b.imports, [1, 2]);
});

test("a purchase is never assumed: reloading a locked page does not resume", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  const loadsBefore = b.opened.length;
  b.reload(chapterUrl(1));
  await new Promise((r) => setTimeout(r, 50));
  // Resuming would pause again on the same locked page, so the status alone
  // proves nothing. What must not happen is the job acting on its own.
  assert.equal(b.opened.length, loadsBefore + 1, "the job reloaded the chapter by itself");
  assert.equal((await b.job()).status, "paused");
});

test("a challenge pauses, and completing it resumes on its own", async () => {
  const b = makeBrowser({ plan: planFor(1, 2), pages: { [chapterUrl(1)]: "challenge" } });
  await b.send("start", { seriesId: "s" });
  const paused = await b.until("paused");
  assert.equal(paused.pause.reason, "challenge");

  b.pages[chapterUrl(1)] = "ok";
  b.reload(chapterUrl(1));
  await b.until("done");
  assert.deepEqual(b.imports, [1, 2]);
});

test("a redirect away from the chapter pauses", async () => {
  const b = makeBrowser({
    plan: planFor(1),
    pages: { [chapterUrl(1)]: { redirectTo: "https://accounts.example/login" } },
  });
  await b.send("start", { seriesId: "s" });
  const paused = await b.until("paused");
  assert.equal(paused.pause.reason, "redirected");
  assert.deepEqual(b.imports, []);
});

test("a server error pauses rather than failing every chapter", async () => {
  const b = makeBrowser({ plan: planFor(1, 2), importStatus: () => 500 });
  await b.send("start", { seriesId: "s" });
  const paused = await b.until("paused");
  assert.equal(paused.pause.reason, "server");
  assert.equal(paused.current.no, 1);
});

test("the user can pause and continue", async () => {
  const b = makeBrowser({ plan: planFor(1, 2, 3) });
  await b.send("start", { seriesId: "s" });
  await b.send("pause");
  const paused = await b.until("paused");
  assert.equal(paused.pause.reason, "user");
  await b.send("resume");
  await b.until("done");
  assert.deepEqual(b.imports, [1, 2, 3]);
});

test("a plan with a foreign URL is refused before any page opens", async () => {
  const plan = planFor(1, 2);
  plan.chapters[1].url = "https://evil.example/content/s/2";
  const b = makeBrowser({ plan });
  const reply = await b.send("start", { seriesId: "s" });
  assert.equal(reply.ok, false);
  assert.match(reply.error, /not a ReadToon chapter page/);
  assert.deepEqual(b.opened, []);
});

test("only one job at a time", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  const second = await b.send("start", { seriesId: "s" });
  assert.equal(second.ok, false);
  assert.match(second.error, /already running/);
});

test("a page on another origin cannot start a job", async () => {
  const b = makeBrowser({ plan: planFor(1) });
  const evil = { id: RUNTIME_ID, frameId: 0, url: "https://evil.example/", tab: { url: "https://evil.example/" } };
  const reply = await b.send("start", { seriesId: "s" }, evil);
  assert.equal(reply.ok, false);
  assert.deepEqual(b.opened, []);
});

test("the VoxNovel page can, but not from inside a frame", async () => {
  const page = { id: RUNTIME_ID, frameId: 0, url: `${SERVER}/series/s`, tab: { url: `${SERVER}/series/s` } };
  const b = makeBrowser({ plan: planFor(1) });
  assert.equal((await b.send("start", { seriesId: "s" }, page)).ok, true);
  await b.until("done");

  const framed = { ...page, frameId: 3 };
  assert.equal((await makeBrowser({ plan: planFor(1) }).send("start", { seriesId: "s" }, framed)).ok, false);
});

test("an unknown command is refused", async () => {
  const b = makeBrowser({ plan: planFor(1) });
  const reply = await b.send("navigate", { url: "https://evil.example/" });
  assert.equal(reply.ok, false);
  assert.deepEqual(b.opened, []);
});

test("a malformed series id is refused", async () => {
  const b = makeBrowser({ plan: planFor(1) });
  const reply = await b.send("start", { seriesId: "../evil" });
  assert.equal(reply.ok, false);
});

test("a finished job's result can be dismissed", async () => {
  const b = makeBrowser({ plan: planFor(1) });
  await b.send("start", { seriesId: "s" });
  await b.until("done");
  const reply = await b.send("clear");
  assert.equal(reply.ok, true, reply.error);
  assert.equal(await b.job(), null);
  assert.equal(b.store.voxJob, undefined);
});

test("a live job cannot be dismissed, only cancelled", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  const reply = await b.send("clear");
  assert.equal(reply.ok, false);
  assert.equal((await b.job()).status, "paused", "a paused job lost its place");
});

test("the job tab closes once the job is done", async () => {
  const b = makeBrowser({ plan: planFor(1, 2) });
  await b.send("start", { seriesId: "s" });
  await b.until("done");
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(b.closed.length, 1);
  assert.deepEqual(b.openTabs(), []);
});

test("skipping the last chapter finishes the job and closes its tab", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  assert.deepEqual(b.closed, [], "a paused job's tab must stay open for the user");
  await b.send("skip");
  await b.until("done");
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(b.closed.length, 1);
});

test("a cancelled job keeps its tab", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "login" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  await b.send("cancel");
  await new Promise((r) => setTimeout(r, 20));
  assert.deepEqual(b.closed, []);
});

test("a tab the user has taken over is not closed", async () => {
  const b = makeBrowser({ plan: planFor(1), pages: { [chapterUrl(1)]: "locked" } });
  await b.send("start", { seriesId: "s" });
  await b.until("paused");
  // The user wanders off to another site in that tab, then skips.
  b.openTabs()[0].url = "https://news.example/";
  await b.send("skip");
  await b.until("done");
  await new Promise((r) => setTimeout(r, 20));
  assert.deepEqual(b.closed, []);
});
