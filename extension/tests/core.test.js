// Run with: node --test extension/tests/
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../lib/core.js");

const NOW = 1_000_000;

function plan(nos, extra = {}) {
  return {
    series_id: "s",
    series_title: "Series",
    already_imported: 0,
    chapters: nos.map((no) => ({
      no,
      url: `https://readtoon.com/content/s/${no}`,
      title: `ตอนที่ ${no}`,
      is_locked: false,
    })),
    ...extra,
  };
}

function run(job, ...events) {
  return events.reduce((j, e) => core.reduce(j, e, NOW), job);
}

const outcome = (o, detail) => ({ type: "outcome", outcome: o, detail });

test.describe("chapter URLs a job may open", () => {
  test("accepts ReadToon chapter pages", () => {
    for (const url of [
      "https://readtoon.com/content/my-novel/12",
      "https://www.readtoon.com/content/my-novel/12",
      "https://readtoon.com/content/my-novel/12.5",
      "https://readtoon.com/content/my-novel/12/",
    ]) {
      assert.equal(core.isReadtoonChapterUrl(url), true, url);
    }
  });

  test("refuses anything else", () => {
    for (const url of [
      "http://readtoon.com/content/s/1", // not https
      "https://evil.example/content/s/1",
      "https://readtoon.com.evil.example/content/s/1", // suffix trick
      "https://evilreadtoon.com/content/s/1", // no dot before readtoon
      "https://readtoon.com/content/s", // series page, not a chapter
      "https://readtoon.com/account/settings",
      "https://user:pw@readtoon.com/content/s/1",
      "https://readtoon.com:8443/content/s/1",
      "javascript:alert(1)",
      "not a url",
      undefined,
    ]) {
      assert.equal(core.isReadtoonChapterUrl(url), false, String(url));
    }
  });

  test("one bad URL rejects the whole plan", () => {
    const p = plan([1, 2]);
    p.chapters[1].url = "https://evil.example/content/s/2";
    assert.match(core.validatePlan(p), /not a ReadToon chapter page/);
  });

  test("a well-formed plan passes", () => {
    assert.equal(core.validatePlan(plan([1, 2, 3])), null);
  });

  test("a plan without numbers or chapters is refused", () => {
    assert.ok(core.validatePlan({}));
    const p = plan([1]);
    p.chapters[0].no = "1";
    assert.ok(core.validatePlan(p));
  });
});

test.describe("the series a page belongs to", () => {
  test("from a series page and from a chapter page", () => {
    assert.equal(core.readtoonSeriesId("https://readtoon.com/content/my-novel"), "my-novel");
    assert.equal(core.readtoonSeriesId("https://readtoon.com/content/my-novel/12"), "my-novel");
    assert.equal(core.readtoonSeriesId("https://www.readtoon.com/content/my-novel/"), "my-novel");
  });

  test("nothing from other pages or sites", () => {
    for (const url of [
      "https://readtoon.com/",
      "https://readtoon.com/content/",
      "https://evil.example/content/my-novel",
      "https://readtoon.com/content/../etc",
      "chrome://extensions",
      undefined,
    ]) {
      assert.equal(core.readtoonSeriesId(url), null, String(url));
    }
  });
});

test.describe("reading a chapter page", () => {
  test("text wins, even over an invisible challenge widget", () => {
    assert.equal(
      core.classifyPage({ paragraphCount: 40, characterCount: 9000, challengeVisible: true, timedOut: false }),
      "ok"
    );
  });

  test("a purchase prompt is reported at once", () => {
    assert.equal(core.classifyPage({ bodyText: "... ยืนยันการซื้อตอน ...", timedOut: false }), "locked");
    assert.equal(core.classifyPage({ bodyText: "เหรียญไม่เพียงพอ", timedOut: false }), "locked");
  });

  test("a login prompt is reported as login, not as a purchase", () => {
    assert.equal(core.classifyPage({ bodyText: "เข้าสู่ระบบเพื่อซื้อ ยืนยันการซื้อตอน", timedOut: false }), "login");
  });

  test("a challenge gets the full wait, since it may clear itself", () => {
    assert.equal(core.classifyPage({ challengeVisible: true, timedOut: false }), "pending");
    assert.equal(core.classifyPage({ challengeVisible: true, timedOut: true }), "challenge");
  });

  test("nothing recognisable is pending, then empty", () => {
    assert.equal(core.classifyPage({ bodyText: "loading", timedOut: false }), "pending");
    assert.equal(core.classifyPage({ bodyText: "loading", timedOut: true }), "empty");
  });
});

test.describe("the job", () => {
  test("imports every chapter in order, then finishes", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("imported"), outcome("imported"));
    assert.equal(job.status, "done");
    assert.deepEqual(job.imported, [1, 2]);
  });

  test("a single chapter is an ordinary job", () => {
    const job = run(core.createJob(plan([7]), NOW), outcome("imported"));
    assert.equal(job.status, "done");
    assert.deepEqual(job.imported, [7]);
  });

  test("an empty plan is done before it starts", () => {
    assert.equal(core.createJob(plan([]), NOW).status, "done");
  });

  for (const reason of ["locked", "login", "challenge", "server", "redirected"]) {
    test(`pauses on ${reason}, on the same chapter`, () => {
      const job = run(core.createJob(plan([1, 2]), NOW), outcome(reason));
      assert.equal(job.status, "paused");
      assert.equal(job.pause.reason, reason);
      assert.equal(job.pause.no, 1);
      assert.equal(core.current(job).no, 1, "a pause must not move past the chapter");
      assert.ok(job.pause.message);
    });
  }

  test("continue retries the same chapter", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("locked"), { type: "resume" });
    assert.equal(job.status, "running");
    assert.equal(core.current(job).no, 1);
    assert.equal(job.pause, null);
  });

  test("skip records why and moves on", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("locked"), { type: "skip" });
    assert.equal(job.status, "running");
    assert.equal(core.current(job).no, 2);
    assert.deepEqual(job.skipped, [{ no: 1, reason: "locked" }]);
  });

  test("skipping the last chapter finishes the job", () => {
    const job = run(core.createJob(plan([1]), NOW), outcome("locked"), { type: "skip" });
    assert.equal(job.status, "done");
  });

  test("skip does nothing unless paused", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), { type: "skip" });
    assert.equal(core.current(job).no, 1);
    assert.deepEqual(job.skipped, []);
  });

  test("an empty page is retried once, then skipped", () => {
    let job = run(core.createJob(plan([1, 2]), NOW), outcome("empty"));
    assert.equal(core.current(job).no, 1, "first empty page should be retried");
    job = run(job, outcome("empty"));
    assert.equal(core.current(job).no, 2);
    assert.equal(job.skipped[0].no, 1);
  });

  test("the retry count starts over for the next chapter", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("empty"), outcome("imported"), outcome("empty"));
    assert.equal(core.current(job).no, 2, "chapter 2 got only one empty result");
  });

  test("outcomes are ignored while paused", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("challenge"), outcome("imported"));
    assert.equal(job.status, "paused");
    assert.deepEqual(job.imported, []);
  });

  test("cancel stops a running or paused job, and is final", () => {
    for (const first of [[], [outcome("locked")]]) {
      const job = run(core.createJob(plan([1, 2]), NOW), ...first, { type: "cancel" }, { type: "resume" });
      assert.equal(job.status, "cancelled");
    }
  });

  test("a user pause keeps the place", () => {
    const job = run(core.createJob(plan([1, 2]), NOW), outcome("imported"), { type: "pause" });
    assert.equal(job.status, "paused");
    assert.equal(job.pause.reason, "user");
    assert.equal(core.current(job).no, 2);
  });
});

test.describe("resuming on its own", () => {
  const url = "https://readtoon.com/content/s/1";

  test("a challenge or login resumes when that chapter loads again", () => {
    for (const reason of ["challenge", "login"]) {
      const job = run(core.createJob(plan([1]), NOW), outcome(reason));
      assert.equal(core.shouldAutoResume(job, url), true, reason);
      assert.equal(core.shouldAutoResume(job, `${url}/`), true, "trailing slash");
    }
  });

  test("buying a chapter never resumes on its own", () => {
    const job = run(core.createJob(plan([1]), NOW), outcome("locked"));
    assert.equal(core.shouldAutoResume(job, url), false);
  });

  test("a load of some other page does not resume", () => {
    const job = run(core.createJob(plan([1]), NOW), outcome("challenge"));
    assert.equal(core.shouldAutoResume(job, "https://readtoon.com/content/s/2"), false);
    assert.equal(core.shouldAutoResume(job, "https://readtoon.com/login"), false);
  });

  test("a user pause is never undone by a page load", () => {
    const job = run(core.createJob(plan([1]), NOW), { type: "pause" });
    assert.equal(core.shouldAutoResume(job, url), false);
  });
});

test.describe("the payload sent to the server", () => {
  test("matches what the one-click importer already sends", () => {
    const payload = core.buildImportPayload({
      url: "https://readtoon.com/content/s/1", seriesId: "s", seriesTitle: "Series",
      chapterNo: 1, chapterTitle: "ตอนที่ 1", paragraphs: ["a"], coverUrl: null,
    });
    assert.deepEqual(Object.keys(payload).sort(), [
      "chapter_no", "chapter_title", "cover_url", "paragraphs", "series_id",
      "series_title", "source", "source_language", "url",
    ]);
    assert.equal(payload.source, "readtoon");
  });
});
