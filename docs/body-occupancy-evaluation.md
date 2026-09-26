# Body occupancy and speaker evaluation

## Contract

`Paragraph.speaker` identifies the mind responsible for dialogue or thought.
An occupancy period maps that mind to a physical body from an inclusive
`(chapter_number, paragraph_index)` position to an optional exclusive end.
Dialogue uses the body's voice; thought uses the mind's own voice. Narration
uses the narrator voice unless the paragraph has an explicit voice override.
An unresolved speaking body falls back to the narrator and is shown as needing
review. Old `inhabiting`/`inhabiting_from_chapter` records load as one open-ended
period.

The registry editor adds and removes periods. The speaker review page displays
the resolved voice and allows an exceptional voice override per paragraph.

## Deterministic checks

Run `.venv/bin/python -m unittest discover -s tests`. The occupancy tests cover
the paragraph immediately before, at, and after each start/end, recurrence,
reciprocal swaps, displaced minds, invalid overlaps, legacy loading, prompt
context, review editing, and the voice actually passed to TTS. These checks do
not require an external model or full audio generation.

For an audible spot check, configure distinct reference voices for mind and body,
then synthesize a short chapter with dialogue and thought on both sides of a
boundary. Listen to the four paragraph WAV files and confirm the two dialogue
voices switch at the configured index while thoughts retain the mind's voice.

## Real-text evaluation

The existing stored review labels cover chapter 68 (42/45 speaker guesses
correct) and chapter 169 (49/49). Those scores test the old name-attribution
task. They are **not** evidence that the classifier identifies a mind under a
body-name alias or that the swap voice is correct. Chapter 169 has no saved
occupancy start, and its reviewed labels need checking under the stricter
"speaker means mind" definition before using it as swap gold data.

Build a gold set with passages before, during, and after an actual swap. Record
the exact paragraph boundaries, speech type, responsible mind, and expected
voice for every spoken or thought line. Include paragraphs whose prose uses
the body's name, the mind's name, an unattributed quote, and first-person
narration. Hold out one passage from any prompt revisions. Run the current
classifier on the gold set before and after a prompt change, then compare mind
accuracy, final voice accuracy, and narrator fallback rate. The release
criterion is no known wrong voice after review and no regression on ordinary
chapters.
