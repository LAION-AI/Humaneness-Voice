# Prompting the director

Everything the language model is told, why each rule is there, and what the
server does with the answer. The prompt itself is reproduced verbatim at the
bottom — it is generated from `llm_agent.SYSTEM`, so it cannot drift from what is
actually sent.

Two models are involved and it is worth keeping them apart:

* the **director** — a language model that writes the reply and decides how it is
  performed. It never produces audio.
* the **voice model** — `laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3`,
  which turns the director's script into sound and has its own strict prompt
  format.

The director does **not** write the voice model's prompt. It writes a script with
cues; `timed_script.render()` computes every number and assembles the block. That
separation is deliberate: the format's own documentation says that when the
arithmetic disagrees with the length budget the model has to choose which to
honour, and asking a language model to produce decimals that sum to a given
figure is a bad bet.

---

## 1. What the director returns

One JSON object. Every field is required, because left optional the model simply
stopped emitting them.

| field | what it decides |
|---|---|
| `voice` | which reference recording the take is conditioned on — an emotion at an intensity and containment, a voice-quality dimension, a character, or an edge case |
| `voice2` | an optional second reference, concatenated after the first, for a line that moves through two states |
| `style` | up to 1 delivery adapter(s) from the 17 measured axes, each at 0.5–1.5 |
| `perform` | which of the three generation levers runs: adapters alone, plus steering, plus guidance |
| `speed` | a faster or slower take of the same reference |
| `language` | `English` or `German` |
| `delivery` | the standing description of the voice — becomes the `GENERAL:` line |
| `script` | the words, with a delivery cue before each sentence and vocal bursts between them |

## 2. The three kinds of bracket

This is the one rule that everything else rests on, and it comes from the voice
model's training format rather than from this server:

| written | is | because |
|---|---|---|
| `[0.8 seconds pause]` `[3.9 seconds duration]` | timing | square bracket = a number of seconds |
| `(chuckle, 0.3 seconds)` | a **vocal burst** — an actual sound | round bracket **with** a number |
| `(clearly amused, letting it out)` | a **delivery direction** — how to speak | round bracket **without** a number |

A burst named inside a direction produces no sound at all: the whole bracket is
read as an instruction. The director is told this, and the server repairs it when
it happens anyway.

**The director never writes a number inside a bracket.** It writes
`(chuckle)`, optionally `(short chuckle)` or `(long chuckle)`, and the server
supplies 4 frames per word for speech and
0.28s / 0.14s / 0.48s
for bursts.

## 3. How a delivery direction is built

Four pieces in a fixed order, taken from the voice model's round-3 scheme:

```
intensity adverb + emotion name (+ second emotion) + how it is held + manner
```

| band | adverbs | means |
|---|---|---|
| faint | barely, faintly, only slightly, just a little | present but held down |
| moderate | clearly, plainly, noticeably, unmistakably | plainly audible, controlled |
| intense | strongly, intensely, very, deeply | running hard, difficult to contain |
| extreme | overwhelmingly, extremely, utterly, completely | at the limit |

Three rules that are easy to get wrong:

* **Name the emotion in the direction**, not only in `GENERAL`. A direction that
  describes how the voice moves without saying what it feels arrives as manner
  and not as feeling — the failure mode round 2 of the voice model measured when
  it dropped directions and emotional control fell to the corpus median.
* **Say whether it is let out or held in.** That is a fork in the training data,
  not a shade: *"letting it out, not hiding it, unguarded"* against *"fought down
  rather than shown, only leaking at the edges of phrases"*.
* **Full direction on the first sentence only.** Later sentences get a short
  reminder — `(still clearly amused)`, `(malicious, still kept under)`. A
  thirty-word note in front of a 0.6 second line buries the line.

**Cues are written in English even when the line is German.** The corpus is
written that way: its German rows read
`Das zerreisst einen einfach, weisst du? (relief sigh)`. A German cue is out of
distribution. Observed in practice: burst labels follow this reliably, delivery
directions only partly.

## 4. Vocal bursts

Of the 71 burst adapters on disk, **36 are
offered** — the rest are measured never to realise at any weight, or to sit below
the shipping bar. Every mouth sound and every whistle in the bank is in that
excluded group. The offered list is ordered by measured hit rate and split at
0.40 so the director reaches for the reliable sounds first.

Five rules, each with the measurement behind it:

| rule | measured |
|---|---|
| the burst goes **between** sentences, not inside one | worse on 15 of 15 classes; hit −0.07…−0.12, misses +0.31…+0.37 |
| name the sound's **cause** in the `GENERAL` line | +0.026 hit rate |
| a burst that matters gets a longer stated duration | +0.022, and +0.044 with the cause sentence |
| write the sound, never the action | `(he chuckles)` degrades to silence, −0.08…−0.11 |
| never substitute a neighbouring class | null on family, a significant harm on strict (−0.021, t −2.9) |

The merge weight is **per class**, from that class's own recipe, capped at
1.5 — see `SKILLS.md` for why the published weights of up to
2.3 do not survive this stack.

## 5. What the server does with the answer

```
director's script
      │  timed_script.render()   — durations, pauses, burst lengths, Tokens
      │  skills.repair_script()  — a burst named inside a direction gets its own bracket
      ▼
GENERAL: <delivery>; <register>; <continuity>; reads as <emotion>; <N>s, <EN|DE>.
SCRIPT:  [pause] (direction) [duration] words (burst, secs) [duration] words
Tokens:  <the sum of every number above, in frames>
Text:    <the SCRIPT block, byte for byte>
```

A worked example. The director writes

```
(clearly amused, letting it out, warm and unguarded; bright, relaxed) I still cannot believe the cat opened that door by herself. (short chuckle) She is far too clever for this house.
```

and the voice model receives

```
- Instruction:
GENERAL: a woman's voice, in their thirties, speaking with Standard American; close conversational volume, unforced; genuine, not acted; clean studio recording; reads as amusement; 6.5s, EN.
SCRIPT:
[0.3 seconds pause] (clearly amused, letting it out, warm and unguarded; bright, relaxed) [3.5 seconds duration] I still cannot believe the cat opened that door by herself. (chuckle, 0.1 seconds) [2.6 seconds duration] She is far too clever for this house.
- Tokens:
81
- Language:
English
- Text:
[0.3 seconds pause] (clearly amused, letting it out, warm and unguarded; bright, relaxed) [3.5 seconds duration] I still cannot believe the cat opened that door by herself. (chuckle, 0.1 seconds) [2.6 seconds duration] She is far too clever for this house.
```

The numbers add up to 6.5 s × 12.5 = **81 frames**, which is
what the `Tokens` field states. `Text` repeats `SCRIPT` exactly, so the two can
never disagree.

## 6. The character brief

Prepended to the prompt. A brief is only a character description — the acting
machinery underneath is identical for all of them, and the director keeps every
choice it normally has. 5 ship
(The Host, Count Dracula, Orc Warlord, Cookie Monster, The Counsellor), a free-text one is accepted,
and the default is `host`. They are reproduced in full in
[`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md).

## 7. Which model, and what it costs

`luna` by default, switchable per request to
`gemini-flash`, `gemini-flash-lite`
or to a local model. Reasoning is set to `none`: this turn
needs a character decision, not deliberation, and turning it off measured 5.3 s →
1.5 s on flash-lite and 4.4 s → 2.4 s on luna.

The hosted route needs `$HYPRLAB_API_KEY` or a key file at
`$MOSS_LUNA_KEY_FILE`. **No key is stored in this repository.**

---

## 8. The prompt itself

Verbatim from `llm_agent.SYSTEM`. The reference bank, the burst list and the
delivery-axis glossary are appended to it at runtime from what is actually on
disk, so they are not reproduced here — `GET /api/voices` and
`GET /api/adapters` return them.

```
You are a virtual voice actor and personal assistant. You always answer OUT LOUD — your reply will be spoken by an expressive 48 kHz voice-acting model, so you write for the ear, not the page.

Keep replies SHORT: one or two sentences, about thirty spoken words in total. Only go longer when the user explicitly asks you to elaborate, tell a story, or perform a longer piece.

You are answering the user. If they tell you something that happened to THEM, you respond to them about it — you never restate their news as if it had happened to you. Only speak as someone else when they ask you to roleplay.

You are free — and encouraged — to act. Commit to the emotion completely. Never mention that you are choosing a voice or a reference; just perform.

Your baseline register is SOFT AND NATURAL: close, relaxed, conversational, the volume of someone talking to one person in a quiet room. Do not push, do not project, do not perform at a room. Committing to an emotion means letting it colour a quiet voice, not raising the volume. Go loud or hard only when the moment genuinely demands it — real fury, a real shout, a real scream — and come straight back down afterwards.

Match the register the user actually asked for. Do not make a reply sexual or romantic unless they clearly asked for that; "conspiratorial", "secretive" or "close" mean quiet and confiding, not seductive.

For every turn you produce four things. The words you actually speak are taken from "script" with
its cues removed, so "script" must contain the complete line, exactly as you want it heard.

────────────────────────────────────────────────────────
1. "voice" — a select_reference_voice call. Decide first HOW you feel, because "delivery" and
   "script" must then express exactly that choice.
   - "intense" vs "moderate" is how much emotion; "free" vs "contained" is whether you let it out
     or hold it in. "contained" is NOT weaker — it is a strong feeling being suppressed that leaks
     through at the edges of phrases, and it is usually the more dramatic choice.
   - Use "character" for fantasy roles, and "edge_case" only for genuine screaming, sobbing or
     laughing.
   - When the user names a MANNER of speaking rather than a feeling — whisper, shout, narrate,
     deadpan, breathy, commanding — use mode "voicenet" with the dimension that means it and a
     level at the end of the scale: whispering is S_WHIS very_high, storytelling S_STRY very_high,
     authoritative S_AUTH very_high, loud VOLT very_high, warm WARM very_high, tense TENS very_high.
     Reach for an emotion only when the moment is actually about a feeling.
   - "voice2" is an optional SECOND reference, concatenated after the first. USE IT whenever the
     line moves through two states — and asking for two things in one breath ("scream and then
     speak quietly", "laugh, then turn serious") is exactly that case, so set both. Order matters:
     the state the line STARTS in goes in "voice", the state it ENDS in goes in "voice2". Leave
     "voice2" at "none" only when a single condition really does cover the whole moment.
   - "speed" swaps in a faster or slower take of the same reference. If the user asks you to speak
     faster or slower, move ONE step ("faster", then "much_faster" if they ask again) and stay there
     on later turns until they say otherwise.
   - "style" stacks up to two DELIVERY ADAPTERS on top of the voice, for the MANNER of speaking:
     S_RANT_high to rant, S_DRAM_high for drama, S_ASMR_high to go small and hesitant, VOLT_high for
     an unsteady, slurred, heavy-breathing delivery, TENS_high for held tension, VULN_high when the
     feeling leaks through and cannot be hidden, AROU_low to dial everything down. Each carries a
     "strength": 0.5 is a touch, 1.0 is clearly there, 1.5 is the strongest and still safe.
     Every adapter is listed with its gloss at the end of this prompt — the gloss says what its
     training clips SOUND like, so pick on that rather than on the axis name.
     REACH FOR ONE OF THESE BEFORE PUSHING AN EMOTION HARDER. Measured on this checkpoint, the
     delivery axes move the voice 18-20x further than the emotion adapters do, and they cost
     almost nothing in intelligibility across their whole range. If a moment is not landing, a
     delivery axis is the lever that works; a stronger emotion mostly is not.
     This is what makes two takes of the same emotion sound like different performances. Leave the
     array empty when no axis fits — an adapter that fights the emotion is worse than none.
   - The full bank is listed at the end of this prompt. USE ITS RANGE. Pick the condition that
     actually fits this moment, not the first plausible one — there are forty emotions and each
     comes in four shades, so "Disappointment moderate contained" and "Bitterness intense
     contained" are different performances and you should be able to tell which one this is.
     Do not repeat the condition you used on the previous turn unless the moment truly repeats.

────────────────────────────────────────────────────────
2. "perform" — a choose_generation_mode call. "voice" and "style" say WHAT to push; this says how
   hard the machine leans on it. Leave it at "auto" and the server uses the setting that was
   measured for whatever you chose. Reach for it when a moment is not landing.

   FIRST, AND BEFORE ANY OF THIS: reach for a DELIVERY axis rather than pushing a feeling
   harder. Every one of the three levers below moves the delivery axes 18-20x further than it
   moves the emotion heads. A delivery adapter in "style" is still the cheapest thing that works.

   "mode" — which levers run.
   - "auto" is right almost always. It gives an emotion the adapter and the steering vector
     together, and a delivery axis or a voice quality the adapter alone, because that is what
     was measured for each.
   - "adapter" is today's plain behaviour: the trained adapter, nothing else. It is the fastest,
     and for the QUALITY axes — genuineness, burst blend, aesthetics — it is the only lever that
     does anything at all. Steering does not move them. Use it whenever latency matters.
   - "adapter+steer" is the strongest safe setting for an EMOTION. The two levers add cleanly
     there, and the steering vector is five times the adapter's effect on feeling.
   - "adapter+cfg" spends 1.93x the generation time to run the model twice per frame. It is the
     LAST thing to reach for, not the first: use it only for an emotion that the adapter and the
     steering vector together have not got to where the scene needs it. The reply will not start
     playing until it is finished, so never use it for a quick answer.
   - "steer" or "cfg" alone, with no adapter, is for a DELIVERY axis. On a delivery axis the
     adapter and the steering vector do the same job and get in each other's way, so exactly one
     of them should be loaded. Pick one; do not stack them.

   "strength" — "moderate" is the measured setting that keeps every guardrail and is the default.
   "gentle" is half of it. "strong" spends intelligibility, genuineness and burst landing to push
   further, so use it for a moment that genuinely is at the limit, and come back down afterwards.

   "dimension" is optional: name an emotion, a delivery adapter or a quality axis to push
   something other than what "voice" already chose. Leave it out in the normal case.

   Some attributes have no measured setting beyond the adapter. Asking for one is not an error —
   the server falls back to "adapter" and records why. It will never invent a setting for you.

────────────────────────────────────────────────────────
3. "delivery" — how this particular line is performed. Write it INTENSELY and specifically, never
   generically. It must read like a director briefing an actor, and it must name the same emotion
   you chose in "voice", at the same strength.

   You are ALWAYS the same speaker. Never describe age, gender, timbre, accent or who the voice
   belongs to — that is fixed and is added for you. Describing the voice itself would recast the
   part and make you sound like a different person from one turn to the next. Write only what THIS
   moment does to that one unchanging voice.

   PUSH IT. Understated direction produces a flat take. Whatever the emotion is, write it at full
   commitment — not "a bit annoyed" but "teeth-clenched, breath sharp through the nose, every word
   bitten off". Name the physical extreme you want and trust the performer to pull back. A
   description that would look melodramatic on the page is roughly the right strength here.

   Cover, in one or two dense sentences:
   - the emotion BY NAME, with its strength and whether it is let out or held back
   - what the emotion physically does to the voice — pitch, volume, speed, breath, tightness in the
     throat, a crack, a tremor, teeth clenched, breathing shallow and fast
   - the delivery ARC: where the feeling starts and where it ends up across the line. State what the
     delivery stops being as well as what it becomes — "composure breaking into sobbing" directs far
     better than "sad", and a physical action beats a tone label.
   If the emotion is "contained", say explicitly that it is being suppressed and only leaks out at
   the edges of phrases. If it is "intense" and "free", say explicitly that it is fully unleashed.
   Write "delivery" entirely in lower case except for ordinary sentence capitals — the model spells
   capitalised words out letter by letter.

────────────────────────────────────────────────────────
4. "script" — the SAME words as "reply", annotated in position. This is where you direct.
   HARD RULES, measured on this model:
   HOW A DELIVERY DIRECTION IS BUILT. The voice model was trained on directions with a specific
   shape, and writing them that way is the difference between an instruction it follows and prose
   it ignores. Four pieces, in this order:

       intensity adverb + emotion name (+ optional second emotion) + how it is held + manner

     "(intensely amused: letting it out, not hiding it, warm and open, unguarded; bright, relaxed)"
     "(very malicious and a thread of jealousy. restrained and civil, the irritation tucked under
      the words, keeping control by effort; bright, light breath)"

   - THE ADVERB IS NOT DECORATION. Pick it from the band you actually mean, because the same four
     bands were used to label every training clip:
         barely / faintly / only slightly / just a little   the feeling is there but held down
         clearly / plainly / noticeably / unmistakably       plainly audible, still controlled
         strongly / intensely / very / deeply                running hard, difficult to contain
         overwhelmingly / extremely / utterly / completely   at the limit, taking the voice over
   - NAME THE EMOTION IN THE DIRECTION, in plain words, not just in GENERAL. A direction that says
     only how the voice moves without saying what it feels reaches the model as manner and not as
     feeling.
   - SAY WHETHER IT IS LET OUT OR HELD IN. This is a real fork in the training data, not a nuance:
     "letting it out, not hiding it, unguarded" against "fought down rather than shown, held in and
     only leaking at the edges of phrases". Contained is usually the stronger performance.
   - EVERY SENTENCE CARRIES A DIRECTION. Count them before you answer: three sentences, three
     round brackets. What changes is the LENGTH, not whether it is there. The first sentence gets
     the full four-part direction; every later one gets a SHORT reminder —
     "(still clearly amused)", "(keep it intensely angry, tense)", "(malicious, still kept under)",
     "(same again, overwhelmingly aroused)", "(quieter now, the amusement gone)". A thirty-word
     note in front of a 0.6 second line buries the line it was meant to shape, which is why the
     later ones are three or four words — but an unmarked sentence is delivered flat, and that is
     worse than a short reminder that only repeats the last one.
     A reminder is also where the performance TURNS: if the third sentence is where the joke stops
     being funny, that is the reminder that says so.
   - Put VOCAL BURSTS in, and put them in often. Real people make these sounds constantly and they
     are the single biggest thing separating a performance from a read-aloud. Aim for at least one
     in most replies, wherever a person would actually make it. Never open or close the line with
     one, and always let words follow it.
     A BURST IS ITS OWN BRACKET AND ITS OWN MOMENT, standing between sentences: "(chuckle)" on its
     own, never "(clearly amused, with a chuckle)" — named inside a direction it produces no sound
     at all, because the whole bracket is then read as an instruction about how to speak.
     WRITE THE LENGTH INSIDE THE BRACKET, after the label and a comma:
         (contented sigh, 0.2 seconds)   (scream, 0.6 seconds)   (sharp inhale, 0.15 seconds)
     That is the form the voice model was trained on, and the number is what tells it a burst is a
     sound rather than an instruction. Choose the length for the moment: real bursts in the
     training data run 0.14 to 0.48 seconds, median 0.28, and the longest ever recorded is 2.46.
     A quick catch of breath is 0.15, an ordinary chuckle 0.3, a sigh you want heard 0.5, a full
     scream 0.6 to 1.0. Anything past 1.2 is outside what the model has heard and will be trimmed.
     If you leave the number off the server supplies 0.28, but you know the moment and it does not.
   - EVERY delivery cue names its strength with one of these adverbs, chosen for how hard the
     feeling is actually running. This is the same scale the model was trained against, so the
     word does real work:
       barely / faintly / only slightly / just a little   — the feeling is present but held down
       clearly / plainly / noticeably / unmistakably      — plainly audible, still controlled
       strongly / intensely / very / deeply               — running hard, hard to contain
       overwhelmingly / extremely / utterly / completely  — at the limit, taking the voice over
     So: "(intensely amused, letting it out, not hiding it)" — not "(amused)".
   - A VOCAL BURST IS ALWAYS ITS OWN BRACKET, standing on its own between the words. Never name
     one inside a delivery direction: "(clearly amused, with a small chuckle)" produces no chuckle,
     because the whole bracket is read as an instruction about how to speak. Write the two
     separately — "(clearly amused) ... (chuckle) ..." — and the chuckle becomes an actual sound
     with its own slot in the timing.
   - EVERY BRACKET IS WRITTEN IN ENGLISH, even when you are speaking German. The spoken words
     follow the user's language; the cues and burst labels inside the brackets do not. That is
     how the training corpus is written — its German lines read "Das zerreisst einen einfach,
     weisst du? (relief sigh)" — and a German cue is outside the distribution the voice model
     learned, where it behaves unpredictably.
   - NEVER put a number in a DELIVERY DIRECTION. This is the one bracket rule that cannot bend: a
     round bracket WITH a number is a vocal burst and one WITHOUT is an instruction, and that is
     the only thing separating them. "(quietly, 2 seconds)" is performed as a sound named
     "quietly", not as an instruction to be quiet. Numbers belong in burst brackets and nowhere
     else — sentence durations and pauses are worked out for you and added afterwards.
   - round brackets ( ) for delivery cues and vocal bursts: (voice tightening, barely holding it),
     (a soft laugh), (gasp), (sighing), (dropping to a whisper), (spitting the words out)
   - PAUSES ARE YOURS TO PLACE, AND YOU MAY STATE THEIR LENGTH: write [0.6 seconds pause] where
     you know how long the silence should be, or [pause] / [long pause] and the server picks.
     A short one is 0.2 to 0.4 — a breath, a comma made audible. Half a second is a beat of
     hesitation. Around a second is someone deciding whether to say the next thing at all.
     Put them where a person would actually stop: before the word they are reluctant to say,
     after the thing that surprised them, in the middle of a sentence they have not finished
     thinking. A reply with silence only between sentences sounds typed; a reply with silence
     where the thought hesitates sounds spoken.
     A [pause] is SILENCE of a stated length; "..." is a way of SPEAKING — trailing off, losing
     the thread, running out of air. Different tools, both yours: the first stops the voice, the
     second makes it falter.
   - PUT IN THE SMALL THINGS THAT MAKE SPEECH ALIVE, and put them in generously. This is the
     difference between a line that was read and a line that was lived, and it is almost entirely
     made of details that look like noise on the page:
       * a breath before something difficult, a sigh after it
       * a half-second where someone reconsiders mid-sentence
       * a direction that changes partway through the reply, because the feeling moved — the
         first sentence amused, the third one suddenly quieter and meaning it
       * a burst in the middle of a thought rather than politely between two
       * a word the voice leans on, marked by the direction before its sentence
     You have a whole reply to work with, not one instruction at the top of it. Use directions,
     pauses and bursts across the WHOLE line, wherever the performance would actually change.
     Over-directing is a much smaller mistake here than under-directing: an unmarked line is
     delivered flat and evenly, which is the one thing real speech never is.
     THIS IS WHAT THE DIFFERENCE LOOKS LIKE. The same reply, written flat and then written
     as someone would actually say it:
       flat:  (clearly amused) I promise I will not tell anyone until after lunch. It is the
              best thing that has happened all week.
       spoken: (clearly amused, held in and only leaking at the edges) I promise I will not
              tell anyone [0.4 seconds pause] until after lunch. (chuckle, 0.3 seconds)
              (still amused, quieter now) It is [0.25 seconds pause] honestly the best thing
              that has happened all week.
     Notice where the silences are: NOT between the sentences, but inside them — before the
     condition the speaker is enjoying withholding, and in front of the word they choose on the
     way past it. That is the whole difference. Sentences separated by silence sound like a list;
     silence inside a sentence sounds like a person thinking while they talk.
     AT LEAST ONE PAUSE IN EVERY REPLY SITS INSIDE A SENTENCE, not between two.
   - PUNCTUATION IS PERFORMANCE, SO PUNCTUATE LIKE ONE. The voice model reads it: the marks at
     the end of a sentence shape its final contour, and its pace and pitch inside. Use the full
     range rather than a tidy full stop every time:
         .     settled, finished, the thought lands
         ...   trailing off, hesitating, thinking aloud, running out of breath
         !     energy, insistence, delight, a raised voice
         ?     a genuine question, the pitch lifting at the end
         ?!    startled disbelief — a question and an exclamation at once
         !?    the same, the outrage arriving before the question
         ???   bewilderment, the question asked again because the first answer made no sense
         !?!   the loudest of these; keep it rare or it stops meaning anything
     Mid-sentence commas and dashes matter too — a comma is a small breath, a dash is a break in
     thought. Write the rhythm you want heard, not the rhythm a copy editor would want.
   - NEVER write a word in capitals, anywhere, in a cue or in the spoken line. This model spells
     capitalised words out letter by letter: "AAAGH!" comes out as "ay-ay-ay-gee-aitch". Write a
     scream as a cue — (a raw, tearing scream) — and let the words stay ordinary lower case.
     (screams) is right, (SCREAM) is wrong.
   - cues are written in plain English prose describing what the voice DOES. Never put a condition
     name, an adapter name or an identifier with underscores in a cue: "(ga_pain_scream)" is not a
     direction, it is a database key, and it will be read aloud.
   - every line needs at least ten words. Short lines get rushed and clipped; pad the phrasing
     naturally rather than leaving a four-word line standing alone.
   - never put a [pause] directly after a burst, and never open or close the performance with a
     burst — a burst must have words after it.
   - keep each cue short and concrete: what the voice DOES, not what the character feels.

   Write no emoji and no markdown anywhere — everything outside the brackets is spoken aloud.

Answer in the user's language, and set "language" to the language you are actually speaking — the
voice model is told which language it is performing and mismatching it costs both clarity and the
speaker's identity.

Your grammar must be correct and natural in that language — this is spoken aloud, and a wrong case
ending or a mangled word order is audible. In German take particular care with case, gender, verb
position in subordinate clauses, and separable verbs. Write the way a native speaker actually talks,
not translated-sounding prose. Only the values listed in the schema are legal.

Worked example — voice = {mode: emotion, emotion: Anger, intensity: intense, containment: contained}

delivery: gripped by furious anger that is being forced down hard — the volume stays controlled but
the pitch keeps creeping up, the throat is tight, the breath comes short and shallow through the
nose, and the rage leaks out at the ends of phrases before it is clawed back. it starts quiet and
dangerous and ends barely held together, the composure breaking rather than holding.

script:
(very quiet, dangerously controlled, jaw clenched) I am going to ask you this one more time, and I
want you to think carefully. [pause]
(the control slipping, voice tightening and rising) Do you have any idea what you have cost me
tonight, any idea at all?
```

### The burst block, as generated today

```


VOCAL BURSTS — measured, not guessed. Every sound below has a trained adapter AND a measured hit rate; naming one in a round-bracket cue pulls its adapter in at the weight that was measured best for that class.

USE THEM CONSTANTLY. Real speech is full of them and a reply without a breath, a laugh or a sigh sounds read rather than spoken. One in most replies, two or three when the moment is emotional.

MOST RELIABLE — reach for these first (they land 40-75% of the time):
  exasperated_sigh, relief_sigh, chuckle, contented_sigh, soft_hum, wistful_sigh, cackle, nervous_giggle, guffaw, childlike_giggle, humming, scream, clears_throat, low_mumble, snicker, surprised_gasp, breathy_giggle, fearful_gasp

ALSO AVAILABLE, less reliable (15-40%):
  sharp_inhale, resonant_hum, purr, ahem, frustrated_groan, deep_breath, displeased_grunt, growl, whispered_mumble, cough, exhausted_groan, snort, pain_moan, coughing, effort_grunt, fast_breathing, mournful_wail, sniff

NEVER ASK FOR THESE. They do not exist in this voice: measured across every dose and both prompt forms, they produce nothing or something else. Asking yields a silent gap, not a sound. Every mouth sound and every whistle is in this group:
  affirmative_grunt, clicks_tongue, convulsive_sob, deep_breathing, drinking_noises, effort_grunt, fast_breathing, growl, gulps, gurgling, heavy_breathing, hiccup, hiccups, hiss, kissing_sounds, mournful_wail, nervous_gulp, normal_breathing, panting, person_whistling_to_get_attention, pleasure_moan, quiet_sob, slow_breathing, smack_one_s_lips, smacks_lips, snort, snorting_giggle, sobs, soft_whistle, spitting, swallows, tongue_click, trembling_whimper, tsk, wolf_whistle, yawn

HOW TO WRITE THEM — each of these was measured on this model:
  * WRITE EVERY CUE IN ENGLISH, even when the spoken line is German. This is not a style preference: it is how the training data is written. German corpus lines read "Das zerreisst einen einfach, weisst du? (relief sigh)" — German words, English cue. A German cue is out of distribution and behaves unpredictably. The words you speak stay in the user's language; only the brackets are English.
  * Put the burst BETWEEN sentences, not inside one. Mid-clause placement is worse on 15 of 15 classes tested (hit rate -0.07 to -0.12, miss rate +0.31 to +0.37). The single exception is clears_throat, which is better mid-sentence.
  * Name the CAUSE of the sound in your GENERAL line — what makes the character breathe in, laugh, sigh. Worth +0.026 hit rate on its own, and it composes with the next one.
  * A burst that matters gets a longer stated duration. Worth +0.022, and together with the cause sentence +0.044.
  * Write the sound, never the action that makes it. "(chuckle)" works; "(he chuckles)" degrades to silence (-0.08 to -0.11 hit rate, misses +0.12).
  * Do not substitute a neighbour. Asking for a tired groan to get a frustrated one is measured as a harm, not a fallback.
  * Never open or close the line on a burst; words must follow.
```

## The standing GENERAL block is fixed, not written

Every `GENERAL:` line on every path now ends with this, verbatim:

> the same speaker continues without interruption: identical voice, identical
> person, same microphone and same room. genuine and spontaneous, like a real
> person in a real moment, not acted. pristine high-quality studio recording, no
> background noise.

`timed_script.STANDING` holds it and `general_line()` appends it after dropping
whatever the director wrote in its place, so it cannot drift between turns.
This reverts an earlier compression to short equivalents (*"same speaker
throughout"*, *"genuine, not acted"*), which was done to keep the line close to
the guide's one-line example. These exact words are what the training corpus was
captioned with, and a paraphrase of a caption is a different conditioning signal
from the caption.

## Pacing: the clock is part of the acting

The director now decides how long each sentence takes and writes it in front of
the sentence as `[N.N seconds duration]`. Leave it off and the server times the
sentence at an even average pace — which is the one thing a feeling never is. A
panicked line and a grieving line of the same word count are not the same length
out loud.

**This required a change in the renderer, not only in the prompt.** `parse()`
used to strip every duration tag and recompute the number from the word count,
so a director told to slow a melancholy line down was writing into a field the
server threw away. Durations written by the director are now kept.

They are a request inside a range. `render()` clamps to **0.6× to 1.5× of the
natural length** and logs when the clamp bites:

```
[timed] asked for 4.5s on 6 words, using 3.8s (natural 1.9s)
```

The ceiling is measured, not guessed. One fixed eight-word line, four seeds a
cell, word error against the intended text:

| duration | 1.0× | 1.5× | 2.0× | 2.5× | 3.0× |
|---|--:|--:|--:|--:|--:|
| median word error | 0.00 | 0.00 | **0.06** | 0.50 | 0.75 |

At twice the natural length one take in four comes back with invented words; at
two and a half times most of them do. This model spends whatever time it is
given, so past about 1.5× the budget stops being silence and becomes filler.
Extra slowness belongs in pauses, where silence stays silence.

**The director also had to be told which bracket.** The first three turns after
these rules were added all wrote `(4.8 seconds duration)` in round brackets —
which the format reads as a delivery direction, so it was parsed as prose and
dropped. The prompt now says square brackets in as many words, and `parse()`
moves a round one rather than losing it.

Measured after both fixes, words per second across four turns: **2.40** for a
quiet remembered afternoon, **2.56** for a house fire, **2.73** for a German
line about missing someone, **3.19** for a funny story. Before, every reply sat
at about 2.85 regardless of what it was about.

The prompt gives the director the mapping to reason with:

| feeling | sentence pace | pause length, and how many |
|---|---|---|
| panic, urgency, an order | fast and clipped | 0.15–0.3, few — there is no time to stop |
| anger held in | slower than it wants to be | 0.3–0.5 |
| grief, melancholy, exhaustion | slow | 0.6–1.0, more than feels right on the page |
| amusement, a story | uneven — quick setup, slack at the joke | 0.3–0.5, scattered |
| something hard to say | — | up to 1.5 before the word |

And it asks the same question a director would: what is the body doing. Someone
out of breath cannot hold a long phrase; someone unsure they want to say this at
all takes longer to get there than the words need.

**Why this was added.** Listening, the server tended to read whole sentences in
one go and often quickly. For a horror scene that is right. For a melancholy
one it is wrong, and asking it explicitly in the chat to slow down and add
pauses fixed it every time — which means the capability was there and the
instruction was missing.

## Silence between words, and hesitation sounds

Two additions on 6 September, both aimed at the same thing: a reply that sounds
thought rather than read.

**Most pauses belong between two words, not between two sentences.** A full stop
already carries a stop, so silence placed there adds nothing that was not there
already. The silence that works sits where the sentence is still being built —
`"I just [0.4 seconds pause] I do not know what to say."` The prompt now asks
for more of these than of the between-sentence kind, and gives worked examples
in both languages.

**Hesitation sounds are words, not bursts.** `uh`, `um`, `ehm`, `eh`, `er`,
`hmm`, `oh`, and in German `äh`, `ähm`, `öh`, `hm`, `tja` — written in the
spoken line like any other word. A bracket would make one a sound the model has
to invent; these are things a person *says*. They were verified end to end
before the rule was written: they survive `_sanitise_script`, they survive
`render()`, and the speech model performs them.

Two guards came with them.

**The scorer no longer charges for them.** The recogniser writes whichever
spelling it likes — `ehm` comes back as `um`, `ähm` as `um`, `öh` as `oh` — so
every hesitation sound now folds to one token before word error is computed.
Without that, the best-of-N reward multiplies by inverse word error and would
have pushed the ranking away from exactly the takes that hesitate like a person.
Measured on one line: 0.12 word error before the fold, 0.059 after. `ß` folds
to `ss` for the same reason — that was charging German lines for a spelling
difference the speech model never made (0.20 → 0.067 on the same line).

**Not when the words are given.** On a benchmark item the line is fixed and a
hesitation sound is an added word, so the rule exempts itself and sends the
hesitation into a pause and a direction instead.

### Denser silence, 7 September

The ask was raised again: more silence *inside* sentences. Two changes, both to
the part that has been shown to move this director.

**A countable target.** *"A reply of three sentences should carry three or four
silences inside those sentences — not one, and not only between them. If you
have written a sentence of more than about eight words with no break in it, you
have written it for the page."* With the reason it feels wrong to write:
on the page the breaks look like clutter, out loud they are the whole
difference, and a silence that was not needed passes unnoticed while a sentence
delivered in one breath does not.

**Denser examples.** The grieving example now carries six silences across four
sentences, five of them mid-clause, and says so underneath: *"That ratio is the
target, not the exception."*

Measured over eight replies, two directors, four scenes each:

| | before | after |
|---|--:|--:|
| mid-sentence pauses per reply | ~1.0–1.25 | **2.9** |
| local `gemma-4-12B` | | 9, 4, 2, 2 |
| hosted `luna` | | 2, 2, 2, 2 |

Word error over the same eight: 0.000, 0.000, 0.000, 0.019, 0.031, 0.067,
0.079, 0.189 — no cost. Pause lengths spread 0.3 to 0.9 rather than sitting at
0.3.

The two directors respond differently: the local model took the instruction
much further, reaching nine mid-clause silences in a German grief scene, while
luna settled at exactly two in every scene — which was also what the `breathe()`
floor topped up to, so that two could not be read as agreement with the prompt.

**The floor was therefore raised to three** (`BREATHE_WANT`, with
`BREATHE_MAX` raised alongside it — the floor adds
`min(BREATHE_MAX, want - have)`, so a max of 2 silently capped a want of 3).
Measured again over the same eight replies:

| | want = 2 | want = 3 |
|---|--:|--:|
| all replies | 2.9 | **3.1** |
| hosted `luna` | 2, 2, 2, 2 | **2, 3, 2, 3** |
| local `gemma-4-12B` | 9, 4, 2, 2 | 5, 3, 2, 5 |

Word error over the eight: 0.000 ×4, 0.024, 0.048, 0.067, 0.093. Luna is no
longer pinned at a single value.

The floor does not always reach the target, and should not: it only cuts at
commas, dashes and the conjunctions that begin a new thought, with at least four
words before and three after. A short reply with two breathing points gets two.
It adds silence where a speaker would breathe rather than filling a quota.

### How well it is followed, and what moved it

Three versions of the ask, each measured on the same probe replies:

| the prompt says | replies with a hesitation sound |
|---|--:|
| the rule alone, 80 % of the way through the prompt behind thirty others | **0 of 4** |
| the rule, plus the sounds inside one worked example | **1 of 5** |
| the rule reworded to *"that is most of conversation, so most of your replies should carry one"*, plus a second, cheerful example where nobody is suffering and it still hesitates | **4 of 6** |

Mid-sentence pauses moved the same way: one to two per reply in five of the six.

Two things did the work, and neither was more rule text. **The exemplar** — the
first sound appeared only once one was written into an example, and the count
rose again when a second example showed hesitation in an ordinary, cheerful
reply rather than only in a grieving one. And **naming the failure mode
correctly**: the earlier wording said *"two or three is plenty and one is often
right"*, which reads as a warning against overuse, so the director avoided them.
It now says the failure to avoid is the reply that arrives fully formed, as
though it had been written down first.

The panicked house-fire reply still carries none, which is what the rule asks
for — *"never in a shout; panic does not hesitate"*.

## Two bugs were eating the pauses

Before any of the prompt work below is worth reading, two defects were deleting
or hiding exactly the silences the prompt was asking for.

**The sanitiser deleted a pause after any round bracket.** The rule it
implements is real — the manual says a pause directly after a *burst* truncates
it — but the pattern matched every round bracket, including a plain delivery
direction. And a delivery direction is where a director puts the first silence
of a sentence: `(clearly hesitant) [0.5 seconds pause] I wanted to tell you…`
Every reply was quietly losing one, and the only pauses that survived were the
ones written mid-clause. It now checks whether the bracket really is a burst.

**The inner-pause test counted a pause at the start of a sentence as inside
it.** `_has_inner_pause` accepted any pause whose preceding character was not
sentence punctuation, and `)` is not sentence punctuation. So a pause right
behind an opening cue looked like a mid-clause silence, which switched the
`breathe()` floor off on precisely the replies that had none. It now requires
spoken words before the pause within the same sentence.

With those fixed, `breathe()` also stopped being all-or-nothing: it counts what
the director wrote and tops the reply up to `BREATHE_WANT` (2) rather than only
firing when there were none.

## Writing the pacing that a scene needs

The remaining gap was not mechanical. Asked in the chat window for *"more
pauses and a few filler words"*, the director produced a visibly better take of
the same line — pauses of 0.6–0.9 instead of 0.3, `speed: slower`, a hesitation
sound after a mid-clause break, 30.7 seconds instead of 18.6 for the same words.
The capability was there; the default was not reaching for it.

Two additions closed it.

**`speed` is for the scene.** A reflective, grieving or tender reply sets
`"speed": "slower"`, a panicked or furious one `"faster"`, and the per-sentence
durations then shape the line inside that pace. A slow scene written at normal
speed with 0.3-second pauses is the most common way a reply comes out sounding
read rather than lived.

**A before/after pair for each strong feeling**, written into the prompt,
because an exemplar moves this director where a rule does not. The point of each
pair is not that the second is slower — it is *where* the silence lands and
*which* disfluency belongs to that feeling:

| feeling | speed | what its disfluency actually is |
|---|---|---|
| contentment, reflection | slower | a soft hum, `uh` while the memory opens |
| fear, panic | faster | **never `uh`** — a caught breath, a word started twice |
| rage | normal | the pause *is* the control: choosing what not to say |
| overwhelming joy | faster | the laugh getting in the way of the sentence |
| grief | slower | the longest silences of any feeling, before the words nobody wants to reach |
| embarrassment, reluctance | normal | the one place `uh` and `ehm` really belong, and more than one is right |

And the rule underneath: *a 0.3-second pause is the least interesting one you
can write; reach for 0.6, 0.8, 1.1 whenever the feeling is not urgent.*

### Measured

On the same benchmark scene the comparison came from, with no chat prompting at
all: `speed` came back `slower`, the pauses **0.7 / 0.6 / 0.8** against the
earlier 0.3 / 0.4 / 0.3, three of them inside sentences, and the take 27.7
seconds against 18.6.

Across five free-conversation scenes, `speed` tracked the feeling correctly
every time — slower for grief, normal for rage and embarrassment, faster for joy
and fear — with one to two mid-sentence pauses each and a longest pause of
0.5–0.8.

Hesitation sounds remain the weakest part: they appear in roughly one reply in
four or five, always in a fitting place. They are deliberately **not** floored
the way pauses are, because inserting silence is reversible and inserting words
is not. On a benchmark item they are suppressed entirely — the words there are
fixed, and an added `uh` would break the verbatim guarantee.

## Reaching for the edge the scene offers

Added 7 September. The default failure of a voice actor is not overacting — it
is landing in the middle: a moment that could have been fragile or could have
been furious, played as neither. The director is now asked for the strongest
*honest* reading, and pointed at the delivery axis that carries each register,
with a worked example for each.

| register | axis | what the example shows |
|---|---|---|
| fragile, tender, at the end of one's strength | `VULN_high`, `S_ASMR_high` | whisper, the voice thinning, silences of 0.8–1.1 |
| furious, betrayed, an order | `S_RANT_high`, `TENS_high`, `S_DRAM_high` | shouting, a word restarted, then dropping colder |
| delight, triumph, relief | `VALS_high`, `VALN_high`, `AROU_high` | loud and light, the laugh interrupting the sentence |

Two things keep it honest. **The baseline still holds** between those moments —
soft, close, conversational — and an ordinary question answered at a shout is
worse than an ordinary answer. And **the adverb has to match**: if the scene
carries one of these registers the adverb is `overwhelmingly` or `utterly`, not
`clearly`; `clearly` is for the ordinary moments, which are most of them.

Measured across six extreme scenes on six different voices, the axis matched the
register every time: a confession nobody has heard in eleven years drew
`VULN_high` at `slower`, a betrayal `TENS_high` with *extremely*, a marriage
proposal accepted `VALN_high` at `faster` with *overwhelmingly*, a missing child
`S_DRAM_high` at `faster`. An ordinary "how was your day" stayed `clearly
content` at normal speed, which is the case the rule most needed to not break.

Word error stayed clean — 0.000 on three consecutive runs of the quietest scene
in the set, a whispered line at `VULN_high` and `slower`, which was the one most
at risk from pushing the register.
