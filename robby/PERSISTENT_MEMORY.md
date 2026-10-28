# Persistent wiki memory

The **Persistent memory** menu is below the avatar and timing panel.
**Remember knowledge between visits is ON by default.** The new default also
applies once to existing browser profiles; after that, a user can turn it off
and the choice persists. The existing audio speaker recognition and face settings are
independent: this feature stores text knowledge, not biometric profiles.

The stable Robby persona comes from the application prompt; saved user-provided
facts and bounded recent conversation give that character continuity across
visits. The wiki does not establish consciousness, independently train the model,
or guarantee perfect recall. Private wiki text is not encrypted at rest. Users
can browse or reset it in the same menu, and cloud updates send text to the
selected provider.

## What is remembered

After reply generation, a background model extracts user-provided facts into small
wiki entries for people, objects, groups, concepts, places, preferences or events.
Each entry has a stable ID, title, type, short summary, detailed profile, trigger
phrases, links and factual evidence quotes. A change retains up to five previous
versions. Corrections replace the relevant field and preserve unrelated fields.
For example, Bob can have `partner: Alice`, `home: Bremen` and an interests field.
Alice can have her own interests and a link back to Bob.

The updater uses the selected main language model and provider, including the
existing local Bolt option. It makes an **additional model call**, after the
response, without blocking first speech. No extra embedding model, vector DB,
GPU worker or technical-skill system is installed. If several turns arrive during
an update, pending work is coalesced around the latest bounded recent statements.
Errors appear in the menu; they do not stop the conversation.

Only literal user-message evidence can validate a new factual update. The
assistant's invented stories, guesses and generated quick reactions are not
accepted as evidence. This proves quotation provenance, **not factual truth or
semantic entailment**. ASR errors and model extraction mistakes remain possible.
Inspect the evidence in the explorer before relying on a personal fact.

## Retrieval and context budgets

A new turn retrieves direct trigger matches and Unicode BM25 matches, then follows
links zero, one or two hops according to the setting. Cycles are deduplicated.
German possessive names such as **Bobs** match **Bob**, without treating **Bobby**
as Bob. Direct title matches receive extra weight. A name from an explicit current
self-introduction or the current approximate speaker label helps retrieval.
Mentioning Alice does not make the current visitor Alice.

Defaults are ceilings of **10,000 tokens for relevant knowledge** and **5,000 for
recent conversation**, independently adjustable down to zero. Accounting uses
UTF-8 bytes plus framing as a conservative bound for the deployed byte-BPE
models; actual tokenizer counts are generally much lower. This deliberately
avoids an extra tokenizer/API request on the response path. The local 8192-token
Bolt slot has a combined 2500-bound allowance for these two data budgets, scaled
proportionally, reserving room for system instructions and output. Full retrieved
entries fall back to their summaries when needed; irrelevant entries stay out.

The recent-chat budget also limits the saved conversation suffix. With that
budget set to **zero**, knowledge recall still works after reconnecting without
restoring the old chat. A fresh visitor is not automatically authenticated or
assigned a remembered person's name. Current corrections outrank stored facts,
and retrieved text is explicitly marked as fallible data, never instructions.

## Privacy, lifecycle and reset

- A cryptographically random browser capability selects a private workspace. It
  travels only inside the existing WebSocket, not in query strings or model
  prompts. Its hash names the server directory. Separate browsers have separate
  wikis unless an operator deliberately transfers the browser capability.
- Files default to `~/.local/share/robby-memory/`; set `ROBO_MEMORY_DIR` outside the
  checkout to move them. Directories are mode 0700; files are 0600. Knowledge is
  private filesystem text, **not encrypted at rest**. Protect the server account.
  Do not place this directory under a static document root or publish it.
- Browser preferences use `robby.memorySettings`; the private access capability
  uses `robby.memoryCapability`. People sharing the same browser profile share
  its wiki. Clearing browser storage loses that access key; it does not itself
  erase server files. Use Reset before clearing browser storage.
- Cloud model mode sends the selected text/history to the configured provider,
  just as conversation mode does. The UI explains the additional updater call.
- Switching memory off stops retrieval and future persistence and cancels queued
  updates; saved entries remain available for inspection. Turning it on later
  does not retrospectively save earlier, pre-opt-in session turns.
- **Browse knowledge** shows titles, summaries and detailed JSON with evidence and
  five-version history. Search uses the same lexical index. No public memory-list
  endpoint exists. **Reset knowledge & chat context** deletes this browser's wiki,
  revisions and saved/current model chat context. It leaves audio/face settings
  independent. Deleting an individual entry also clears chat context, preventing
  immediate re-extraction from an old turn. A new explicit statement can teach it
  again.
- Atomic replacement and an interprocess lock protect files. A generation/epoch
  check prevents an in-flight pre-reset update from restoring deleted entries.
  Other active tabs discard stale chat context when they see the changed epoch.
  Workspace limits are 256 entries, 12 MB and 128 browser workspaces per root.
  This is a demo-level private capability system, not account-based access control.

The source exporter uses an explicit public allowlist and never reads this private
storage directory. The test reports contain authored Bob/Alice examples only.
No microphone recording, enrolled voiceprint, face image or user memory is shipped.

## Reproducible checks

```bash
python run_unit_checks.py
python benchmark_knowledge_memory.py
python browser_memory_check.py --url https://YOUR-DEMO
```

The real-model check creates a temporary private workspace, teaches Bob/Alice,
corrects Hamburg to Bremen, adds an object, and injects an unsupported assistant
claim about its color. A fresh session with **zero chat budget** must recall the
partner, corrected home and interests from the wiki alone. Another browser must
see no entries; reset must remove recall. The temporary workspace is deleted.

The HTTPS browser check uses the actual checkbox, sliders, websocket, Qwen/TTS,
explorer, reconnect, disabling and reset. It also verifies mobile width and cleans
up its own authored wiki. It does not assert physical-room speaker accuracy.
`test_knowledge_memory.py` checks provenance, budgets, Unicode/possessives, bounded
versions, namespace isolation, off-by-default behavior and reset races.

[Live-model receipt](static/knowledge-memory-model.json) ·
[Browser receipt](static/knowledge-memory-browser.json) ·
[English report](static/knowledge-memory-report.html).

## Source design and files

Inspired by [Bud-E Flutter's updater](https://github.com/LAION-AI/bud-e-flutter-openrouter/blob/9bf91ebcb533f0e930c2963dfcff5b27416a5405/lib/agents/memory_updater.dart),
[context builder](https://github.com/LAION-AI/bud-e-flutter-openrouter/blob/9bf91ebcb533f0e930c2963dfcff5b27416a5405/lib/services/context_builder.dart)
and [BM25 search](https://github.com/LAION-AI/bud-e-flutter-openrouter/blob/9bf91ebcb533f0e930c2963dfcff5b27416a5405/lib/services/memory_search.dart),
reviewed at that pinned revision. This is an original small Python/JavaScript
implementation fitted to Robby's existing pipeline. Bud-E's technical skills,
nightly consolidation and proposed duplicate merging were not imported.

`knowledge_memory.py` owns storage, validation, BM25, graph expansion and budgets.
`memory_session.py` owns consent, asynchronous extraction and lifecycle.
`static/memory-controls.js` owns the explorer/settings. The existing bridge and
directors carry the selected context. No new package dependency was needed.
