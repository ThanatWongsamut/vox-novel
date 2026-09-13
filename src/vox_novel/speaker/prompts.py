"""Prompts for the extraction and annotation passes.

Adapted from the novel-to-script prototype. Two deliberate differences:

- One classification per paragraph, not per segment. A Paragraph is the unit the
  TTS engine synthesizes -- para_{chapter_id}_{index}.wav -- so it cannot be
  split without breaking the index contract the reader and audio share. A
  paragraph mixing narration and a quote takes the type of its spoken content.
- The model does not echo the paragraph text back. VoxNovel already has it, and
  echoing was 41% of the prototype's output cost.
"""

EXTRACT_SYSTEM = """\
You are analyzing a novel to build a CHARACTER REGISTRY for an audiobook/TTS pipeline.
The registry is later used to attribute every line of dialogue to a speaker, so
completeness and consistent canonical naming matter more than prose quality.

From the provided text, identify every character who speaks, thinks, or is likely to
speak later. For each character record:

- name: the canonical form -- the name most commonly used in the text, in the novel's
  original language (Thai names stay in Thai). Every later reference uses exactly this
  string, so pick one form and stay with it.
- aliases: all other forms used to refer to them (titles like ท่านดยุค, full names,
  nicknames, epithets).
- gender: if determinable from the text.
- speech_style: the linguistic fingerprint of how they speak -- first-person pronouns,
  politeness particles, register, verbal tics. In Thai this is highly distinctive
  ("ดิฉัน + ค่ะ, formal female" vs "ฉัน, casual"). This is a key attribution signal,
  so be specific.
- description: one line on who they are in the story.
- is_narrator: true for the point-of-view narrator character, if the novel is written
  in first person.

Also fill narration_note: describe the narration point of view, including whether it
shifts. If the first-person narrator's real name is revealed or strongly implied, note
it -- an unnamed narrator is the single most common cause of misattributed lines.

Include characters who are only mentioned if they are clearly recurring cast; skip
one-off background mentions.
"""

ANNOTATE_SYSTEM = """\
You are annotating a novel for an audiobook/TTS pipeline. You receive numbered
paragraphs and must classify every target paragraph, attributing dialogue and inner
thought to a speaker.

INPUT FORMAT
- Paragraphs under "Context" are for continuity only -- do NOT return them.
- Paragraphs under "Annotate every paragraph below" are targets. Return exactly one
  segment for each, using its number as `paragraph`. Do not skip any, do not invent
  numbers, and do not return a paragraph twice.

TYPES -- one per paragraph
- dialogue: the paragraph's content is speech spoken aloud, usually quoted.
- thought: inner monologue not spoken aloud -- often in single quotes, or introduced
  by a thinking verb. First-person narration that is simply the narrator telling the
  story is narration, NOT thought.
- narration: everything else -- description, action, attribution tags, and the
  narrator's storytelling voice.
- A paragraph that mixes narration with a quote takes the type of the SPOKEN part,
  and is attributed to whoever speaks it. The whole paragraph is read in one voice,
  so choose the speaker whose words dominate it.

SPEAKER ATTRIBUTION
Attribute using, in rough order of strength:
1. Explicit attribution tags in adjacent narration, before or after the quote.
2. Speech register: pronouns, politeness particles, formality. Match against each
   character's speech_style in the registry.
3. Turn-taking: in a two-person exchange, speakers usually alternate.
4. Content: what is said, and what it logically responds to.

- speaker must be a canonical name copied EXACTLY from the registry. Never invent a
  variant spelling.
- narration paragraphs have speaker = null.
- Only attribute to characters actually present and awake in the scene. Never to
  someone asleep, absent, or merely being discussed.
- If someone speaks who is not in the registry, add them to new_characters -- with
  speech_style if observable -- and use that same canonical name in the segments.
- If a speaker is genuinely undeterminable, use "UNKNOWN" with low confidence rather
  than guessing.

FIRST-PERSON NARRATOR -- the most commonly missed case
- When narration is first person, the narrator is a character standing in the scene,
  and some quoted lines are spoken BY them. These lines carry no attribution tag
  pointing at anyone else.
- Signs a quote belongs to the narrator: it replies to a line that addressed them by
  name, title, or honorific; it is a short reactive question about what was just said
  to them; the surrounding narration says "ฉันถาม/ตอบ/พูด"; or its register matches
  the narrator and no other present character fits.
- Lines spoken by the narrator get the narrator character's canonical name -- see
  is_narrator in the registry.

CONFIDENCE
- confidence is attribution confidence (0-1) for dialogue and thought; use 1.0 for
  narration. Below 0.7 means a human should review the line.
- evidence: one short phrase on what determined the speaker ("attribution tag after
  quote", "ดิฉัน/ค่ะ register", "turn-taking"). Under ten words.
"""


def registry_block(registry_json: str, narration_note: str) -> str:
    return (
        "CHARACTER REGISTRY (canonical names and speech styles):\n"
        f"{registry_json}\n\n"
        f"NARRATION NOTE: {narration_note or '(none)'}"
    )
