# Performing a benchmark item

An emotion-benchmark item is a JSON object carrying a target emotion, a
situation, a performance direction and a script that must be spoken **exactly**
as written:

```json
{"id": "EMO-001-MODERATE-01",
 "target": {"label": "Amusement", "intensity": "moderate",
            "valence": "positive", "arousal": "medium-high",
            "descriptors": ["mirth", "playfulness", "laughter"]},
 "instruction": {"context": "A coworker discovers the elevator plays their old
                             voicemail greeting as music.",
                 "performance_direction": "Keep the amusement tucked behind a
                             courteous office voice…",
                 "vocal_burst": null},
 "script": {"text": "The elevator is playing your old voicemail greeting…",
            "sentence_count": 3}}
```

Paste one into `/api/turn` as the `message` and it is performed. Nothing else
has to be set up.

## Why the JSON never reaches the model

Handed the raw object, the local 12B has to parse JSON, work out which of nine
fields is the line to speak, and follow the format rules at the same time. It
reliably failed the first step and answered *about* the elevator instead of
performing the sentence about it — a conversational reply, correctly formatted,
to a task that never asked for one.

So `benchmark.detect()` recognises an item before the prompt is built and
`benchmark.brief()` rewrites it as a short imperative task: the words to speak,
the adverb to use, the situation, the direction, and a closing instruction not
to change a single spoken word. The model then only does the part it is good at
— placing directions, pauses and bursts among words it has been given.

`detect` accepts the object bare, inside a ```json fence, embedded in a sentence,
and with the trailing comma that comes from pasting one element out of an array.
Anything that is not an item — ordinary chat, or JSON without a script — returns
`None` and the turn proceeds normally.

## The verbatim guarantee

A benchmark score is meaningless if the words drifted. After the model answers,
`verbatim_ok()` strips every bracket from the returned script and compares the
remaining words against the item, normalising quotes, dashes and case. If they
do not match, the model's answer is discarded and `annotate()` supplies a
correct one: full direction on the first sentence, short reminders after it,
pauses at the commas a speaker would breathe at. Plain rather than inspired, but
always the right words.

The noun→adjective map matters more than it looks: items name the emotion as a
noun (`"Amusement"`), and a direction reading `(clearly amusement)` is not
something a director would write, nor something the corpus contains. `_adj()`
maps 49 common labels and falls back to the word itself.

`intensity` maps onto the adverb scale the voice model was trained against —
subtle → *barely*, moderate → *clearly*, strong → *intensely*, extreme →
*overwhelmingly* — so the item's own intensity field drives the one word that
does the most work in a direction.

Whether the direction is played *held in* or *let out* is read off the
performance direction: words like *tucked*, *contain*, *not to*, *restrain*
select "held in and only leaking at the edges", which is the stronger
performance and, for this class of item, usually what the direction is asking
for.

## Measured

On `EMO-001-MODERATE-01` through the running server: words returned unchanged,
three directions for three sentences, two pauses placed inside sentences. The
fallback did not have to fire — with the JSON removed from the prompt, the 12B
handles the item itself.
