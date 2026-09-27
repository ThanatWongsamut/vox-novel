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
function makeBrowser({ plan, pages = {}, importStatus = () => 200 } = {}) {
  const store = {};
  const tabs = new Map();
  let nextTab = 1;
  const imports = [];
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
        remove: async (key) => {
          delete store[key];
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
      get: async (id) => {
        const tab = tabs.get(id);
        if (!tab) throw new Error("No tab with id");
        return { ...tab };
      },
      sendMessage: async (id, message) => {
        const tab = tabs.get(id);
        assert.equal(message.action, "EXTRACT_WHEN_READY");
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
    windows: { update: async () => ({}) },
    alarms: { create: async () => {}, clear: async () => {}, onAlarm: event() },
    action: { setBadgeText: () => {}, setBadgeBackgroundColor: () => {} },
    scripting: { registerContentScripts: async () => {}, unregisterContentScripts: async () => {} },
    permissions: { contains: async () => false },
  };

  async function fetch(url, init = {}) {
    const u = new URL(url);
    if (u.pathname === "/api/extension/plan") {
      return { ok: true, status: 200, json: async () => structuredClone(plan) };
    }
    if (u.pathname === "/api/extension/import") {
      const body = JSON.parse(init.body);
      const status = importStatus(body);
      if (status === 200) imports.push(body.chapter_no);
      return { ok: status === 200, status, json: async () => ({ detail: `HTTP ${status}` }) };
    }
    throw new Error(`unexpected fetch ${url}`);
  }

  const context = vm.createContext({
    chrome, fetch, URL, URLSearchParams, console, structuredClone,
    setTimeout, clearTimeout, setImmediate,
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

  return { send, job, until, imports, opened, focused, pages, reload, store };
}

function planFor(...nos) {
  return {
    series_id: "s",
    series_title: "Series",
    already_imported: 0,
    chapters: nos.map((no) => ({ no, url: chapterUrl(no), title: `ตอนที่ ${no}`, is_locked: false })),
  };
}

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
