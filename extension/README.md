# VoxNovel Importer - Chrome Extension

Imports ReadToon chapters into VoxNovel for reading and **VoxCPM2 Thai
audiobook synthesis** -- one chapter with a click, or every missing chapter of a
series as a batch.

ReadToon renders chapters in the browser and gates them behind Turnstile, so
VoxNovel's server cannot fetch them itself. This extension reads them from your
own logged-in Chrome instead, which already passes those checks and holds your
purchases.

---

## Setup

1. Open `chrome://extensions/` in Google Chrome.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select this `extension/` directory.
4. Start VoxNovel: `uv run vox-novel web`

After pulling a new version, press the reload icon on the extension's card in
`chrome://extensions/`, then refresh any open VoxNovel and ReadToon tabs.

---

## One chapter

Open an unlocked chapter on ReadToon and click the floating
**Import to VoxNovel** button, or the extension icon.

## A batch

From VoxNovel's series page, **Import missing chapters** -- optionally with a
chapter range; a range of one chapter imports just that one. The same is in the
extension popup on any ReadToon series or chapter page.

Every ReadToon fetch on the series page -- **Fetch Chapter**, **Re-Fetch**,
**Fetch Next** -- goes through the extension when it is installed, and through
VoxNovel's own headless browser otherwise. Re-fetching keeps the speaker labels
on unchanged paragraphs, including ones you verified. Repeated text needs an
unchanged surrounding sequence; ambiguous lines need review again.

The extension asks VoxNovel which chapters are missing, then loads each one in
a single background tab, waits for the text, imports it, and moves on after 3
seconds. It stops and brings the tab forward whenever ReadToon needs you:

| ReadToon shows | what happens |
| --- | --- |
| the chapter | imported |
| a chapter you have not bought | paused -- buy it and press **Continue**, or **Skip** |
| a login prompt | paused -- log in and press **Continue**, or **Skip** |
| a Turnstile or other check | paused -- complete it; the job carries on by itself |
| a different page | paused -- sort it out and press **Continue**, or **Skip** |
| nothing within 15 seconds | reloaded once, then skipped and listed at the end |

It never solves a check, never buys a chapter, and does not disguise itself as
a person. Closing the job tab or restarting Chrome pauses the job; nothing runs
unattended after a restart.

When a job is done its tab closes -- unless you have since used that tab for
something else. A cancelled job leaves its tab open.
Pause, Skip and Cancel invalidate pending chapter reads. Cancel also aborts a
pending import request, though an import already accepted by the server can
still finish.

**Check ReadToon's terms** before running a batch over a whole novel -- they
may restrict automated copying even of chapters you have bought.

## Security

VoxNovel's page can ask the extension for a job, so the extension trusts
nothing it is sent: it accepts only start, pause, resume, skip, cancel and
status; it fetches the chapter list from the VoxNovel server itself rather than
taking one from the page; and it refuses the whole job if any URL is not a
`https://readtoon.com/content/<series>/<chapter>` page. It does not use the
`debugger` permission.

## Tests

```
node --test extension/tests/
```
