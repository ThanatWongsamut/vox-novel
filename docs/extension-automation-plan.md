# Plan: batch import through the browser

Today the importer takes one chapter per click: open an unlocked chapter in
Chrome, click Import. A 200-chapter novel is 200 manual page loads. This makes
it one action -- "import the missing chapters" -- carried out in the user's own
logged-in Chrome.

## Why the browser at all

ReadToon renders chapters client-side and gates them behind Turnstile, so the
server cannot fetch chapter text itself; the existing headless Playwright path
in `scrapers/readtoon.py` is blocked in practice. The user's own Chrome session
already passes those checks and already holds their purchases.

The series *catalog* is different: `ReadtoonScraper.get_novel_info` fetches it
server-side, with each chapter's number, title and `isPaid`. So the server can
plan the job and the browser only has to load pages.

## What is borrowed from browser-control agents, and what is not

Borrowed: a command channel into the user's browser, so a program -- here the
VoxNovel server or web page -- can say "load this, read it, report back".

Not borrowed:

- **An LLM choosing what to click.** ReadToon's layout is known. A scripted
  driver is faster, free, deterministic and testable.
- **`chrome.debugger`.** It shows the "is being debugged" banner and grants
  control of any page. Content scripts and tab navigation are enough.

## Behaviour

A job is an ordered list of chapter URLs, all `https://readtoon.com/content/...`.
The extension opens one tab and, for each chapter:

1. Navigates to it and waits for the page to settle.
2. Waits up to ~15 s for one of: the chapter text (`div.prose.mx-auto`), a
   purchase/login prompt, or a challenge.
3. Acts on what it found:

| page shows | action |
| --- | --- |
| chapter text | post to `/api/extension/import`, move on |
| purchase prompt (chapter not owned) | **pause** -- buy it and continue, or skip it |
| Turnstile / challenge / login | **pause**, bring the tab to the front, wait for the user |
| nothing within the timeout | retry once, then skip and record it |

A paused job offers Continue (re-read the same chapter) and Skip (record it as
skipped and move on). Buying is always the user's own click on ReadToon.

4. Waits a few seconds before the next chapter.

A paused job resumes when the user clicks Continue, or on its own when the
job tab next finishes loading a chapter page (the usual result of solving a
challenge). It never attempts to solve a challenge, never buys a chapter, and
does not disguise itself as a human -- the delay is there to be polite to the
site, not to evade detection.

Progress lives in `chrome.storage.local`, because MV3 suspends the background
service worker when idle. Every step is resumable from stored state, and a
restart of Chrome resumes the job paused rather than silently.

## Starting a job

**From VoxNovel's series page** -- "Import missing chapters". The page and the
extension share a browser, so a small content script on the VoxNovel origin
relays `window.postMessage` to the extension. No extension ID, no
`externally_connectable`, no new server connection. The page shows live
progress, pause, resume and cancel.

**From the popup** on a ReadToon series or chapter page -- the same job, with an
optional chapter range. A range of one chapter is an ordinary job. The existing
one-click Import button on a chapter page is unchanged.

Both ask the server for the plan:

```
GET /api/extension/plan?series_id=<slug>&from=<n>&to=<n>
-> [{no, url, title, is_locked}]   chapters in range not yet stored
```

The catalog is refreshed first when it is missing or older than the range asked
for.

## Security

The relay turns a web page into something that can drive the browser, so the
background script trusts nothing it is sent:

- Accepts only `start`, `pause`, `resume`, `cancel`, `status`.
- Every URL in a job must parse as `https://readtoon.com/content/<slug>/<n>`
  (or the `*.readtoon.com` equivalent). Anything else rejects the whole job.
- The relay runs only on the configured VoxNovel origin and only accepts
  messages whose `source` is its own window.
- One job at a time.

## Changes

Extension:

- `background.js` (new): the job driver.
- `relay.js` (new): content script on the VoxNovel origin.
- `content.js`: an "extract when ready" message that returns
  `ok | locked | challenge | empty` plus the details it already computes.
- `popup.*`: start, progress, pause/resume/cancel.
- `manifest.json`: a background service worker, the relay content script, and
  `scripting` for registering the relay on a custom server URL.

Server:

- `GET /api/extension/plan`.
- Series page: the button and a progress panel.

## Testing

- Server: the plan endpoint -- range, missing-only, locked flag, traversal.
- Extension: job state transitions, URL validation and page classification are
  pure functions in their own module so they can be tested without a browser.
- Manual: a short range in real Chrome, including one purchased and one
  unpurchased paid chapter, and a pause/resume.

## Decisions

1. Unowned paid chapters pause the job rather than being skipped.
2. 3 seconds between chapters.
3. ReadToon's terms may restrict automated copying even of purchased chapters;
   the user checks before running a whole novel.
