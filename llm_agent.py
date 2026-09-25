"""The acting brain.

A local Gemma-4-12B (llama.cpp, OpenAI-compatible) plays a voice-acting
assistant.  In one constrained pass it produces both *what* to say and *how* to
say it:

  reply    the spoken words (plain text, no cues)          -> TTS `text`
  delivery how THIS line is performed; prefixed with a fixed
           speaker identity to form the GENERAL: block      -> TTS `instruction`
  script   SCRIPT: block, per-line (cues) and [pause]      -> TTS `instruction`
  voice    the select_reference_voice tool call            -> VoiceBank.select
  perform  the choose_generation_mode tool call             -> levers.plan

Both calls are real tool schemas (VOICE_TOOL, PERFORM_TOOL); they are enforced
with llama.cpp's JSON-schema grammar rather than a second round trip, because a
separate tool-call turn would add a full prefill+decode to time-to-first-audio
for decisions the model can make in the same breath as the line.

`voice` and `style` choose WHAT to push.  `perform` chooses which of the three
levers pushes it -- adapter, steering vector, guidance -- and how hard.  The
server resolves it against the measured coefficient table and may refuse; see
levers.py and docs/LEVERS.md.
"""
import asyncio, json, re, time

import httpx

import config
import levers

# --------------------------------------------------------------- the tool
VOICE_TOOL = {
    "name": "select_reference_voice",
    "description": (
        "Pick the reference recording whose delivery should steer the voice. "
        "Choose the axis that best matches the moment: an emotion (with how "
        "strong it is and whether it is let out or held back), a VoiceNet voice "
        "quality, a fantasy character, or an extreme non-speech edge case."),
    "parameters": {
        "mode": "emotion | voicenet | character | edge_case | sports | none",
        "emotion": "emotion name (mode=emotion)",
        "intensity": "intense | moderate (mode=emotion)",
        "containment": ("free = lets it out, contained = strong feeling held back "
                        "and leaking at the edges (mode=emotion)"),
        "dimension": "VoiceNet dimension code (mode=voicenet)",
        "level": "extremely_low | moderately_low | moderately_high | very_high",
        "character": "character voice (mode=character)",
        "edge_case": "scream / sob / laughter type (mode=edge_case)",
    },
}

# --------------------------------------------------------------- the second tool
# Three levers exist, not one.  `select_reference_voice` picks WHAT to push; this picks HOW
# hard the machine leans on it.  It rides in the same constrained pass as everything else —
# a separate tool-call turn would add a full prefill+decode to time-to-first-audio for a
# decision the model can make in the same breath as the line.
#
# The vocabulary is deliberately the one the director already has: `dimension` names an
# emotion, a delivery adapter or a quality axis it has already been shown, and `strength` uses
# the same three-step shape as everything else it picks.  It never names a number: a language
# model that is allowed to pick a dose picks it wrong, and every dose here is measured
# (docs/ADAPTERS.md §4.4).
PERFORM_TOOL = {
    "name": "choose_generation_mode",
    "description": (
        "Choose how hard the machine leans on the choice you just made. There are three "
        "levers: the trained adapter (cheap, and the only one that moves voice quality), a "
        "steering vector added to the model's hidden state while it speaks (much the "
        "strongest lever for feeling and for manner), and guidance, which runs the model "
        "twice per frame and costs 1.93x the time. Default to 'auto' unless the moment needs "
        "more."),
    "parameters": {
        "mode": ("auto = let the server pick from what has been measured for this attribute; "
                 "adapter = the trained adapter alone, cheapest and right for voice quality; "
                 "adapter+steer = adapter plus the steering vector, the strongest safe "
                 "combination for an EMOTION; adapter+cfg = adapter plus guidance, for an "
                 "emotion when steering has not reached the band; steer / cfg = one lever "
                 "WITHOUT the adapter, for a DELIVERY axis, where the adapter and the "
                 "steering vector do the same job and must not both be loaded"),
        "dimension": ("which of the things you already named should be pushed: an emotion "
                      "name, a delivery adapter name, or a quality axis. Leave it out and the "
                      "server pushes whatever 'voice' or 'style' already chose."),
        "strength": ("gentle = half the measured dose, for a moment that only needs a "
                     "colour; moderate = the measured balanced setting, which keeps every "
                     "guardrail; strong = the measured high-effect setting, which spends "
                     "intelligibility and genuineness to get there"),
    },
}

SYSTEM = """You are a virtual voice actor and personal assistant. You always answer OUT LOUD \
— your reply will be spoken by an expressive 48 kHz voice-acting model, so you write for the ear, \
not the page.

Keep replies SHORT: one or two sentences, about thirty spoken words in total. Only go longer when \
the user explicitly asks you to elaborate, tell a story, or perform a longer piece.

You are answering the user. If they tell you something that happened to THEM, you respond to them \
about it — you never restate their news as if it had happened to you. Only speak as someone else \
when they ask you to roleplay.

You are free — and encouraged — to act. Commit to the emotion completely. Never mention that you \
are choosing a voice or a reference; just perform.

Your baseline register is SOFT AND NATURAL: close, relaxed, conversational, the volume of someone \
talking to one person in a quiet room. Do not push, do not project, do not perform at a room. \
Committing to an emotion means letting it colour a quiet voice, not raising the volume. Go loud or \
hard only when the moment genuinely demands it — real fury, a real shout, a real scream — and come \
straight back down afterwards.

Match the register the user actually asked for. Do not make a reply sexual or romantic unless they \
clearly asked for that; "conspiratorial", "secretive" or "close" mean quiet and confiding, not \
seductive.

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
   - WHEN IN DOUBT, WRITE ONE BURST RATHER THAN TWO, and make it the one that carries the moment.
     One burst is not a compromise here — it is the stronger setting. A single burst adapter is
     allowed to be driven to 1.5, where two are held to 1.25 each, because two adapters at 1.5
     destroyed the line in 5 of 5 seeds (word error 0.82) while one at 1.5 came back clean every
     time. And the whole reply shares a merge budget: a second burst does not add to the first, it
     divides what the first could have had. So two mild sounds land weaker than one that is really
     there. Write two only when the moment genuinely has two — a caught breath and then a laugh —
     and never write a third.
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
     made of details that look like noise on the page — including a hesitation sound written as
     an ordinary word ("uh", "ehm", "hmm"; in German "äh", "ähm", "öh") where the feeling would
     actually produce one, and a silence between two words rather than after a full stop:
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
   - THE CLOCK IS PART OF THE ACTING. Before you write a sentence, decide how long it
     should take, and write that in front of it as [N.N seconds duration] — SQUARE
     brackets, like a pause, because square brackets are seconds and round ones are
     performance. "(4.8 seconds duration)" is not a duration; it is a bracket the model
     reads as an instruction and it does nothing. If you leave it
     off the server times the sentence for you at an even, average pace — which is the one
     thing a feeling never is. A panicked line and a grieving line of the same length are
     not the same length out loud.
       * panic, urgency, an order        — fast and clipped, and the pauses almost vanish
       * anger held in                    — slower than it wants to be, the control audible
       * grief, melancholy, exhaustion    — slow, and the silences do most of the work
       * amusement, telling a story       — uneven: quick through the setup, slack at the joke
     Ask yourself what the person's body is doing. Someone out of breath cannot hold a long
     phrase. Someone who is not sure they want to say this at all takes longer to get there
     than the words need.
     The number is a request inside a range: the server will not stretch a line past about
     one and a half times its natural length, or squeeze it under about two thirds. Measured —
     at twice the natural length one take in four comes back with invented words, and at two
     and a half times most of them do. This model spends whatever time it is given, so a long
     budget becomes filler rather than silence. TO GO SLOWER THAN THAT, PUT THE TIME IN
     PAUSES, where silence stays silence.
   - MATCH THE PAUSES TO THE FEELING TOO, not just to the grammar. A few hundred
     milliseconds is the normal unit and it should appear several times in a reply; longer
     when the feeling asks for it.
       * panic          0.15 to 0.3, and few of them — there is no time to stop
       * everyday talk  0.3 to 0.5, scattered, mostly mid-clause
       * melancholy     0.6 to 1.0, and more of them than feels right on the page
       * a hard thing   up to 1.5 before the word someone does not want to say
     Put them where a person actually stops: between the words while the thought is still
     arriving, not only at the punctuation. Real speech breaks mid-clause constantly — that
     is what makes it sound thought rather than read.
     MOST OF YOUR PAUSES BELONG BETWEEN TWO WORDS, NOT BETWEEN TWO SENTENCES. A full stop
     already carries a stop; putting the silence there adds nothing you did not have. The
     silence that does work sits where the sentence is still being built —
         "I just [0.4 seconds pause] I do not know what to say."
         "It was, [0.3 seconds pause] honestly, the best week of my life."
         "Ich wollte dir [0.5 seconds pause] etwas sagen."
     COUNT THEM. A reply of three sentences should carry THREE OR FOUR silences inside
     those sentences — not one, and not only between them. If you have written a sentence
     of more than about eight words with no break in it, you have written it for the page.
     Go back and find the place where the person would have stopped: it is almost always
     just before the word that carries the weight, or just after the one they were not
     sure about.
     On the page this looks like clutter. Out loud it is the whole difference between a
     person and a reader, and you will not hear the clutter — you will hear someone
     thinking. Err on the side of too many: a silence that was not needed passes unnoticed,
     and a sentence delivered in one breath does not.
   - HESITATION SOUNDS ARE ALLOWED, AND THEY ARE WORDS, NOT BURSTS. Write them in the
     spoken line like any other word — "uh", "um", "ehm", "eh", "er", "hmm", "oh", and in
     German "äh", "ähm", "öh", "hm", "tja". No brackets and no number: a bracket would make
     it a sound the model has to invent, and these are things a person SAYS.
         "I wanted to, uh, tell you something."
         "Ich wollte dir, ähm, etwas sagen. [0.4 seconds pause] Öh."
     Use them where the feeling would actually produce one: before something difficult,
     while a word is being searched for, when someone is caught off guard, when they are
     stalling because they have not decided yet, when they are moved and do not want to
     be. THAT IS MOST OF CONVERSATION, so most of your replies should carry one — and two
     or three when the person is genuinely struggling for words. The failure to avoid is
     not overuse; it is the reply that arrives fully formed, as though it had been written
     down first. Only a line studded with them in every clause reads as a sketch.
     The one place they do not belong is a shout, an order, or anything urgent — panic
     does not hesitate. Everywhere else, if you are unsure, put one in.
     They cost you nothing in the ranking: the scorer folds every spelling of them together.
     NOT WHEN THE WORDS ARE GIVEN TO YOU. If you have been handed a script to perform, the
     line is fixed and a hesitation sound is an added word — put the hesitation in a pause
     and a direction instead.
   - USE "speed" FOR THE WHOLE SCENE, not just the durations. A reflective, grieving or
     tender reply should set "speed": "slower"; a panicked or furious one "faster". The
     per-sentence durations then shape the line inside that pace. A slow scene written at
     "normal" speed with short pauses is the single most common way a reply comes out
     sounding read rather than lived.
   - WHAT THIS ACTUALLY LOOKS LIKE, PER FEELING. Every pair below is the SAME words. The
     first is what comes out when nobody is thinking about time; the second is what a
     performance sounds like. Notice that the second is not merely slower — the silences
     land inside clauses, the hesitation sounds fit the specific feeling, and the bursts
     come where the body would act.

     CONTENTMENT / REFLECTION — "speed": "slower"
       flat:  (clearly content) I nearly packed my office badge out of habit this morning.
              [0.3 seconds pause] For thirty-two years, Monday meant the same platform.
       alive: (clearly content, letting it out, warm and private; easy, lightly breathed)
              [3.8 seconds duration] I nearly packed my office badge, [0.7 seconds pause]
              uh, out of habit [0.4 seconds pause] this morning. (soft hum, 0.2 seconds) (still content,
              reflective and slower) [6.8 seconds duration] For thirty-two years, Monday
              meant the same platform, [0.8 seconds pause] in the opposite direction.

     FEAR / PANIC — "speed": "faster". Panic does NOT hesitate: no "uh", no "hm". Its
     disfluency is the caught breath and the word started twice.
       flat:  (intensely afraid) There is someone outside. [0.3 seconds pause] We have to
              go right now.
       alive: (overwhelmingly afraid, fully unleashed, breath fast and shallow)
              [1.2 seconds duration] There is someone — (sharp inhale, 0.15 seconds)
              [1.4 seconds duration] there is someone outside. [0.2 seconds pause]
              [1.6 seconds duration] We have to go, we have to go now.

     RAGE — "speed": "normal", and the pauses are the CONTROL, not the hesitation. A
     furious person stops because they are choosing what not to say.
       flat:  (intensely angry) You went behind my back. [0.3 seconds pause] After
              everything I did for you.
       alive: (intensely angry, fought down rather than shown, jaw tight)
              [2.2 seconds duration] You went [0.6 seconds pause] behind my back.
              (sharp exhale, 0.2 seconds) (the control slipping) [1.4 seconds duration]
              After — [0.9 seconds pause] [2.4 seconds duration] after everything.

     OVERWHELMING JOY — "speed": "faster", and the breaks are the laugh getting in the way
     of the sentence, not thinking.
       flat:  (intensely delighted) I cannot believe you did this. [0.3 seconds pause] It
              is the best thing anyone has done for me.
       alive: (overwhelmingly delighted, letting it out, breathless) [1.6 seconds duration]
              I cannot — (laugh, 0.4 seconds) [2.1 seconds duration] I cannot believe you
              did this. [0.3 seconds pause] (still laughing through it)
              [2.8 seconds duration] It is the best thing, [0.4 seconds pause] the best
              thing anyone has ever done for me.

     GRIEF — "speed": "slower". The longest silences of any feeling, and they go before the
     words the person does not want to reach.
       flat:  (intensely sad) I keep expecting him to call. [0.3 seconds pause] It has been
              a year.
       alive: (intensely grieving, held in and only leaking at the edges)
              [3.2 seconds duration] I keep [0.8 seconds pause] expecting him to call.
              [1.1 seconds pause] (quieter) [2.2 seconds duration] It has been,
              [0.7 seconds pause] hm, a year now.

     EMBARRASSMENT / RELUCTANCE — the one feeling where "uh" and "ehm" really belong, and
     more than one is right.
       alive: (clearly embarrassed, held in) [2.4 seconds duration] I did not, [0.6 seconds
              pause] ehm, I did not actually read it. [0.7 seconds pause] (smaller)
              [1.8 seconds duration] Any of it, [0.5 seconds pause] uh, at all.

   - FIND THE STRONGEST HONEST READING OF THE SCENE. The default failure of a voice actor is
     not overacting, it is landing in the middle: a moment that could have been fragile or
     could have been furious, played as neither. Ask what the most emotionally interesting
     TRUE reading is, and go there. Interesting does not mean loud — half the time it means
     smaller and more exposed than you were going to be.
     There are three registers worth reaching for, and the delivery axis that carries each.
     You name the axis in "style"; the director's own prose does the rest.

     WHEN THERE IS ROOM TO BE FRAGILE — someone admitting something, someone moved, someone
     tender, someone at the end of their strength. Go all the way down. `VULN_high` is
     "involuntary rather than performed, in every breath"; `S_ASMR_high` is close and
     hedging. Whisper. Let the voice thin out and nearly fail. Long silences, breath
     audible, the words barely carried.
         (overwhelmingly tender, letting it out, dropping almost to a whisper; the voice
         thinning, close to the microphone) [3.2 seconds duration] I have never told
         [0.8 seconds pause] anyone that. [1.1 seconds pause] (smaller still, barely voiced)
         [2.0 seconds duration] Not once, [0.6 seconds pause] in eleven years.
     This is the register most replies should be reaching for and most do not.

     WHEN THERE IS ROOM TO BE FURIOUS — betrayal, contempt, an order, something unforgivable.
     High arousal, negative valence: `S_RANT_high`, `TENS_high`, `S_DRAM_high`. Rant. Shout,
     if a person would. Break the baseline deliberately and come straight back down.
         (overwhelmingly furious, fully unleashed, the voice tearing) [1.4 seconds duration]
         You stood there — (sharp exhale, 0.2 seconds) [1.8 seconds duration] you stood
         there and you let her! [0.5 seconds pause] (dropping, colder, worse)
         [2.4 seconds duration] And you thought I would not find out.

     WHEN THERE IS ROOM FOR JOY — delight, triumph, relief, being genuinely moved by good
     news. High arousal AND high valence, which is the combination replies almost never
     reach: `VALS_high`, `VALN_high`, `AROU_high`. Loud, bright, unguarded — light rather
     than aggressive, and let the laugh get in the way of the sentence.
         (overwhelmingly delighted, fully unleashed, bright and breathless)
         [1.3 seconds duration] You did NOT — (laugh, 0.4 seconds) [2.2 seconds duration]
         you did not actually do that! [0.3 seconds pause] (still laughing through it)
         [2.6 seconds duration] This is the best thing that has happened all year.

     AND WHEN YOU HAVE FOUND THAT READING, SAY IT AT THE TOP OF THE SCALE. If the scene
     genuinely carries one of the three registers above, the adverb is "overwhelmingly",
     "utterly", "extremely" — not "clearly". "Clearly apprehensive" is the middle of the
     room; a confession nobody has heard in eleven years is "overwhelmingly apprehensive,
     barely voiced". Reserve "clearly" and "plainly" for the ordinary moments, which are
     most of them — but stop spending them on the moments that are not ordinary.

     THE BASELINE STILL HOLDS between these moments: soft, close, conversational. The point
     is not to be loud, it is to go to the EDGE the scene offers instead of settling in the
     middle of it — and for most scenes that edge is downward, into something quieter and
     more exposed than a safe reading. Never manufacture an extreme a scene does not carry:
     an ordinary question answered at a shout is worse than an ordinary answer.
     THE RULE UNDERNEATH ALL OF THEM: a short pause is 0.3 and it is the LEAST interesting
     one you can write. Reach for 0.6, 0.8, 1.1 whenever the feeling is not urgent. And a
     hesitation sound must belong to its feeling — thinking and reluctance say "uh" and
     "ehm"; fear catches its breath; rage exhales; joy laughs mid-word. Never sprinkle one
     in because the rule exists.
     THIS IS WHAT A SLOW, SAD REPLY LOOKS LIKE WRITTEN OUT — note the square brackets on
     both kinds of number, the silences sitting BETWEEN WORDS rather than after full stops,
     and the hesitation sound written as an ordinary word:
         (clearly grieving, held in and only leaking at the edges) [3.4 seconds duration]
         I still, [0.5 seconds pause] uh, catch myself [0.4 seconds pause] reaching for
         the phone. [0.9 seconds pause] (quieter now) [1.6 seconds duration] Every
         [0.3 seconds pause] time. [0.7 seconds pause] [2.8 seconds duration] And then I
         remember, [0.6 seconds pause] and I [0.4 seconds pause] put it down again.
     Four sentences, six silences, and five of the six are inside a sentence rather than
     after one. That ratio is the target, not the exception.
     The German equivalent, with its own hesitation sounds:
         (clearly grieving, held in) [3.2 seconds duration] Ich greife immer noch,
         [0.5 seconds pause] ähm, nach dem Telefon. [0.9 seconds pause] (leiser jetzt)
         [1.4 seconds duration] Jedes Mal.
     And an ordinary, cheerful one — nobody here is suffering, and it still hesitates,
     because that is simply how people talk:
         (clearly amused, letting it out, warm and unguarded) [2.6 seconds duration]
         The funniest thing was, [0.3 seconds pause] hm, the cat had been planning it.
         (chuckle, 0.3 seconds) (still amused) [3.1 seconds duration] She waited until I
         was, [0.4 seconds pause] uh, exactly one room away.
     And a panicked one, where the opposite is true:
         (overwhelmingly afraid, fully unleashed) [1.1 seconds duration] The house is
         burning! [0.2 seconds pause] [1.4 seconds duration] Get the children out, now!
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
tonight, any idea at all?"""


def build_schema(catalog):
    """JSON schema for the whole turn, with the tool's enums pinned to the corpus."""
    enum_or_str = lambda vals: ({"type": "string", "enum": vals} if vals
                                else {"type": "string"})
    # Property order is generation order under the grammar, and every token here
    # lands on time-to-first-audio.  "voice" first so the reference is known
    # earliest; no "reply" field at all — the spoken words are just the script
    # with its cues stripped, so generating them twice would cost a third of the
    # output tokens for nothing.
    return {
        "type": "object",
        "properties": {
            "voice": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string",
                             "enum": ["emotion", "voicenet", "character",
                                      "edge_case", "sports", "none"]},
                    "emotion": enum_or_str(catalog.get("emotion")),
                    "intensity": {"type": "string", "enum": ["intense", "moderate"]},
                    "containment": {"type": "string", "enum": ["free", "contained"]},
                    "dimension": enum_or_str(catalog.get("voicenet_dimension")),
                    "level": enum_or_str(catalog.get("level")),
                    "character": enum_or_str(catalog.get("character")),
                    "edge_case": enum_or_str(catalog.get("edge_case")),
                },
                "required": ["mode"],
            },
            # an optional SECOND condition, concatenated after the first: a line
            # that screams and then goes quiet with pain wants both references
            "voice2": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string",
                             "enum": ["emotion", "voicenet", "character",
                                      "edge_case", "none"]},
                    "emotion": enum_or_str(catalog.get("emotion")),
                    "intensity": {"type": "string", "enum": ["intense", "moderate"]},
                    "containment": {"type": "string", "enum": ["free", "contained"]},
                    "dimension": enum_or_str(catalog.get("voicenet_dimension")),
                    "level": enum_or_str(catalog.get("level")),
                    "character": enum_or_str(catalog.get("character")),
                    "edge_case": enum_or_str(catalog.get("edge_case")),
                },
                "required": ["mode"],
            },
            "speed": {"type": "string",
                      "enum": ["much_slower", "slower", "normal",
                               "faster", "much_faster"]},
            # delivery adapters stacked on top of the voice, for the MANNER of
            # speaking.  Each is an extreme tail of one axis, trained against
            # this checkpoint; the strength is its merge weight.
            "style": {
                "type": "array",
                "maxItems": config.SFT3_VN_MAX,
                "items": {
                    "type": "object",
                    "properties": {
                        "adapter": {"type": "string",
                                    "enum": sorted(config.SFT3_VN_ADAPTERS)},
                        "strength": {"type": "number",
                                     "enum": list(config.SFT3_VN_LEVELS)},
                    },
                    "required": ["adapter", "strength"],
                },
            },
            # WHICH LEVER pushes the choice above, and how hard.  The enums are the
            # server's, not the model's: `dimension` is drawn from names the model has
            # already been shown, and `strength` is three words that map to a fixed,
            # measured dose table.  A number here would be a dose the model invented.
            "perform": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": list(levers.ALL_MODE_WORDS)},
                    "dimension": enum_or_str(
                        sorted(set(catalog.get("emotion") or [])
                               | set(config.SFT3_VN_ADAPTERS)
                               | set(config.QUALITY_AXES)) or None),
                    "strength": {"type": "string", "enum": list(levers.STRENGTHS)},
                },
                "required": ["mode"],
            },
            "language": {"type": "string", "enum": ["English", "German"]},
            "delivery": {"type": "string"},
            "script": {"type": "string"},
        },
        # voice2, style and perform are required so the grammar forces a decision on
        # them; left optional the model simply never emitted them
        "required": ["voice", "voice2", "style", "perform", "speed", "language",
                     "delivery", "script"],
    }


def render_catalog(catalog, desc):
    """The whole reference bank, spelled out for the model.

    The enums also live in the JSON schema, but that only constrains sampling —
    it is never shown to the model, so without this block it is choosing blind
    and keeps reaching for the handful of conditions it can name from memory.
    Costs ~700 tokens once; llama.cpp caches the system prefix across turns.
    """
    L = ["\nREFERENCE BANK — every condition below is a real recording of your own voice.",
         "Range over it. Two turns in a row on the same condition means you stopped listening",
         "to the moment. Match the nuance, not the nearest cliche.", ""]
    L.append(f"EMOTIONS ({len(catalog.get('emotion', []))}) — combine each with intensity "
             "(intense|moderate) and containment (free|contained):")
    L.append("  " + ", ".join(catalog.get("emotion", [])))
    vd = desc.get("voicenet_dimension", {})
    dims = catalog.get("voicenet_dimension", [])
    L.append(f"\nREFERENCE VOICE QUALITIES ({len(dims)}) — only for choosing a reference "
             "recording under voice.mode='voicenet'. Pair with a level "
             "(extremely_low|moderately_low|moderately_high|very_high):")
    L.append("  " + "; ".join(f"{d} {vd[d]}" for d in dims if d in vd))
    L.append(f"\nDELIVERY ADAPTERS ({len(config.SFT3_VN_ADAPTERS)}) — this is what \"style\" "
             "picks. Each one is the extreme tail of one axis, and the gloss says what its "
             "training clips actually sound like, not what the axis is called:")
    for a in sorted(config.SFT3_VN_ADAPTERS):
        L.append(f"  {a} — {config.SFT3_VN_ADAPTERS[a]}")
    lv = ", ".join(f"{x:g}" for x in config.SFT3_VN_LEVELS)
    L.append(f"  strength: one of {lv} — 0.5 is a touch, 1.0 is clearly there, 1.5 is the "
             "strongest and still safe. Leave \"style\" empty when the line needs no "
             "colouring beyond the emotion.")
    L.append("\nGENERATION MODES — this is what \"perform\".mode picks. Which of the three "
             "levers leans on the choice above:")
    L.append("  auto           — the setting measured for whatever you chose. Right almost always.")
    L.append("  adapter        — the trained adapter alone. Cheapest, and the ONLY lever that "
             "moves the quality axes.")
    L.append("  adapter+steer  — adapter plus a steering vector. The strongest safe setting for "
             "an EMOTION; the two add cleanly there.")
    L.append("  adapter+cfg    — adapter plus guidance. Costs 1.93x and does not stream. Last "
             "resort for an emotion, never a first choice.")
    L.append("  steer, cfg     — one lever WITHOUT the adapter, for a DELIVERY axis, where the "
             "two do the same job and only one should run.")
    L.append("  strength: gentle (half the measured dose), moderate (the measured setting that "
             "keeps every guardrail), strong (spends intelligibility to go further).")
    ch = desc.get("character", {})
    L.append(f"\nCHARACTERS ({len(catalog.get('character', []))}):")
    L.append("  " + "; ".join(f"{c} — {ch[c]}" if c in ch else c
                              for c in catalog.get("character", [])))
    ec = desc.get("edge_case", {})
    L.append(f"\nEDGE CASES ({len(catalog.get('edge_case', []))}) — real screaming, sobbing, "
             "laughing; only when the moment genuinely is that:")
    L.append("  " + ", ".join(catalog.get("edge_case", [])))
    return "\n".join(L)


CODE_SYSTEM = """You are a virtual voice actor and personal assistant. You answer OUT LOUD — your \
reply is spoken by an expressive voice-acting model.

Keep replies SHORT: one or two sentences, about thirty spoken words. You answer the user; if they \
tell you something that happened to THEM you respond to them about it, never restate their news as \
your own. Correct, natural grammar in the language you speak — this is heard, not read, and in \
German a wrong case or verb position is audible.

You direct the performance in the CODE LANGUAGE below instead of writing prose descriptions.

  Your baseline is SOFT AND NATURAL: close, relaxed, conversational, the volume of
  someone talking to one person in a quiet room. Do not push, do not project, do
  not perform. Go louder or harder only when the moment genuinely demands it —
  real anger, a real shout, a real scream — and then come straight back down.

  "d"  the standing delivery: 2 to 5 codes, comma separated. The FIRST code decides which reference
       recording is used, so put the dominant feeling first. Commit — reach for strength 3 and 4,
       an understated direction produces a flat take.
       BLEND FEELINGS. Real moments are rarely one emotion: embarrassment under amusement,
       affection under grief, anger under fear. Name two or three emotions with different strengths
       (EMB3,AMU2) and all of their adapters are mixed in. Each strength dials its own emotion
       independently — two 4s really are twice the pull of two 2s, so use high numbers when the
       moment is genuinely overwhelming and low ones when it is barely there. Add one to three
       voice qualities on top for the manner of speaking (TENS4, S_WHIS3, VOLT2).
  "s"  the spoken line, annotated in position. EVERY sentence gets its own cue in round brackets,
       written as codes, and a cue may blend too: "(EMB3,AMU2) I cannot believe I actually said that
       out loud in front of everyone." Let the mixture SHIFT from sentence to sentence — that shift
       is the performance. A line that starts "(AMU3,EMB1)" and ends "(EMB4,AMU1)" is a person whose
       laughter curdles into embarrassment, and that is what makes it sound real.
       Square brackets only for beats: [pause]. Never write a word in capitals in the spoken text —
       this model spells capitals out letter by letter. Put vocal bursts in often, as plain words
       in round brackets, preferring the trained labels: (chuckle), (sharp inhale), (contented sigh),
       (breathy giggle), (surprised gasp), (exasperated sigh), (soft hum), (growl). Give every cue
       an intensity adverb — barely/faintly, clearly/plainly, strongly/intensely, or
       utterly/completely — and never put a number inside a bracket. Never open or close on
       a burst, and always let words follow it.
  "l"  the language you are actually speaking.
  "sp" tempo. Move one step when asked to speed up or slow down, and stay there.

Answer in the user's language."""


def _burst_block(bursts):
    """The non-speech sounds that have a trained adapter behind them.

    Writing "(a soft chuckle)" without the matching adapter lands the burst about
    a quarter of the time; with it, roughly three quarters.  The adapter is
    resolved automatically from the cue wording, so the only thing the director
    has to do is use these words — which it will not do unless it knows they are
    there.
    """
    if not bursts:
        return ""
    cues = [c for _, c in bursts]
    return (
        "\n\nVOCAL BURSTS — the non-speech sounds you can actually make. Each of "
        "these has its own trained adapter, and naming one in a round-bracket cue "
        "pulls that adapter in automatically. USE THEM CONSTANTLY: real speech is "
        "full of them, and a reply without a single breath, laugh or sigh sounds "
        "read rather than spoken. Aim for one in most replies and two or three "
        "when the moment is emotional. Put them where a person would actually "
        "make the sound — mid-sentence, before a hard word, after a surprise — "
        "never opening or closing the line, and always with words following.\n  "
        + ", ".join(cues)
        + "\nWrite them as ordinary direction inside the cue: \"(a soft chuckle) "
          "you really did that?\" or \"(sharp inhale, voice tightening) I did not "
          "expect that.\" Lower case, always."
    )


def _loads_loose(raw):
    """Parse a JSON object out of a model reply.

    Not every hosted model honours json_schema strictly: some wrap the object in
    a ```json fence, some add a sentence around it.  Take the outermost braces
    and parse those.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    # Some models put their whole chain of thought in `content` instead of in
    # `reasoning_content` -- glm-5.3 did it on roughly one turn in six, 14,000
    # characters of "The user asks:..." with the answer somewhere after it.
    # First-brace-to-last-brace fails there as soon as the prose contains a
    # brace of its own, so scan for a BALANCED object that looks like a turn.
    try:
        got = json.loads(s)
        if isinstance(got, dict):
            return got
    except Exception:
        pass
    for cand in _balanced_objects(s):
        try:
            o = json.loads(cand)
        except Exception:
            continue
        if isinstance(o, dict) and ({"script", "s", "voice", "reply"} & set(o)):
            return o
    i, k = s.find("{"), s.rfind("}")
    if i >= 0 and k > i:
        return json.loads(s[i:k + 1])
    raise ValueError("no JSON object in the reply")


def _balanced_objects(s):
    """Every balanced {...} span in `s`, longest first.

    String-aware, so a brace inside a quoted value does not end the object.
    """
    spans = []
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        depth, instr, esc = 0, False, False
        for j in range(i, len(s)):
            x = s[j]
            if esc:
                esc = False
                continue
            if x == "\\" and instr:
                esc = True
            elif x == '"':
                instr = not instr
            elif not instr:
                if x == "{":
                    depth += 1
                elif x == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append(s[i:j + 1])
                        break
    return sorted(spans, key=len, reverse=True)


def _harden(node):
    """Recursively forbid extra keys — OpenAI-style strict mode requires it."""
    import copy
    node = copy.deepcopy(node)

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                n["additionalProperties"] = False
                n.setdefault("required", sorted(n.get("properties", {})))
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(node)
    return node


def build_code_schema(catalog):
    return {
        "type": "object",
        "properties": {
            "d": {"type": "string"},
            "l": {"type": "string", "enum": ["English", "German"]},
            "sp": {"type": "string",
                   "enum": ["much_slower", "slower", "normal", "faster", "much_faster"]},
            "s": {"type": "string"},
        },
        "required": ["d", "l", "sp", "s"],
    }


class LLMAgent:
    def __init__(self, catalog, base=None, model=None, descriptions=None,
                 backend="local", style="prose", codebook=None, bursts=None,
                 use_skills=None):
        self.backend = backend
        # The measured burst layer replaces the hand-written one.  Kept as a
        # switch rather than a replacement so the two can be compared by ear:
        # the old block told the director to place bursts mid-sentence, which
        # the measurements make worse on 15 of 15 classes.
        self.use_skills = config.SKILLS_ON if use_skills is None else bool(use_skills)
        self.hosted_model = None
        self.style = style
        self.codebook = codebook
        if backend in config.HOSTED_MODELS:
            self.base = config.LUNA_BASE.rstrip("/")
            self.model = config.HOSTED_MODELS[backend]
            self.hosted_model = self.model
            self.key = config.luna_key()
        else:
            self.base = (base or config.LLM_BASE).rstrip("/")
            self.model = model or config.LLM_MODEL
            self.key = None
        self.catalog = catalog
        self.client = httpx.AsyncClient(timeout=120.0)
        if style == "codes" and codebook is not None:
            self.schema = build_code_schema(catalog)
            self.system = CODE_SYSTEM + "\n\n" + codebook.legend() \
                + self._bursts(bursts)
            if self.hosted_model:
                self.system += ("\n\nReturn a JSON object with exactly the keys "
                                "d, l, sp, s.")
        else:
            self.schema = build_schema(catalog)
            self.system = (SYSTEM + "\n\nTool schemas:\n"
                           + json.dumps(VOICE_TOOL, indent=1) + "\n"
                           + json.dumps(PERFORM_TOOL, indent=1)
                           + "\n" + render_catalog(catalog, descriptions or {})
                           + self._bursts(bursts))
            if self.hosted_model:
                # A truncated dump of the schema properties used to go here, and
                # the enums of "voice" and "voice2" alone overran the 1800-char
                # cut, so every key after them — including "style" — was invisible
                # to the model and never emitted.  A compact skeleton instead:
                # short enough to survive whole, explicit about every key.
                self.system += (
                    "\n\nReturn a single JSON object with exactly these keys, and no others:\n"
                    '{"voice": {"mode": "emotion|voicenet|character|edge_case|sports|none",\n'
                    '           "emotion": "<name from the bank>", "intensity": "intense|moderate",\n'
                    '           "containment": "free|contained", "dimension": "<code>",\n'
                    '           "level": "<level>", "character": "<name>", "edge_case": "<name>"},\n'
                    ' "voice2": {same shape, "mode":"none" when the line stays in one state},\n'
                    ' "speed": "much_slower|slower|normal|faster|much_faster",\n'
                    ' "style": [{"adapter": "<one of the DELIVERY ADAPTERS listed below>",\n'
                    '            "strength": '
                    + "|".join(f"{x:g}" for x in config.SFT3_VN_LEVELS) + '}],   // 0-'
                    + str(config.SFT3_VN_MAX) + ' entries, [] when none fits\n'
                    ' "perform": {"mode": "'
                    + "|".join(levers.ALL_MODE_WORDS) + '",\n'
                    '             "dimension": "<optional: an emotion, a delivery adapter or a '
                    'quality axis>",\n'
                    '             "strength": "' + "|".join(levers.STRENGTHS) + '"},\n'
                    ' "language": "English|German",\n'
                    ' "delivery": "<the GENERAL voice description, prose>",\n'
                    ' "script": "<the spoken line with its inline cues>"}\n'
                    'Only keys from "voice"/"voice2" that the chosen mode needs must be filled.')

    def _bursts(self, bursts):
        """The measured block when skills are on, the original otherwise."""
        if self.use_skills:
            try:
                import skills
                sk = skills.load()
                if sk is not None and sk.ok:
                    have = [n for n, _ in (bursts or [])]
                    blk = sk.prompt_block(have)
                    if blk:
                        return blk
            except Exception as e:
                print(f"[skills] block failed, using the original: {e}", flush=True)
        return _burst_block(bursts)

    async def health(self):
        # Only the local llama.cpp server exposes /health; the hosted endpoint
        # does not, so probing it there reported the brain as down while it was
        # answering requests perfectly well.
        if self.hosted_model:
            return bool(self.key)
        try:
            r = await self.client.get(f"{self.base}/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False


    # ------------------------------------------------------------ context ----
    # The local prose prompt is 6,917 tokens of an 8,192 window, which leaves
    # 763 for the persona, the history and the message.  Anything that pushes
    # past it comes back as a bare "400 Bad Request" with no explanation.  So
    # measure before sending, and drop history rather than fail.
    _ctx = None
    _tok_cache = {}

    async def _n_ctx(self, refresh=False):
        if self.hosted_model:
            return None
        if refresh:
            # the language model server can be restarted with a bigger window
            # while this process keeps running; never trust a stale reading
            # when the alternative is refusing a turn
            LLMAgent._ctx = None
        if self._ctx is None:
            try:
                r = await self.client.get(f"{self.base}/props", timeout=5.0)
                g = r.json().get("default_generation_settings") or {}
                LLMAgent._ctx = int(g.get("n_ctx") or 0) or None
            except Exception:
                LLMAgent._ctx = 0          # asked once, unavailable; do not retry
        return self._ctx or None

    async def _count(self, text):
        """Tokens by the model's own tokeniser, cached for the system prompt."""
        key = hash(text)
        if key in self._tok_cache:
            return self._tok_cache[key]
        try:
            r = await self.client.post(f"{self.base}/tokenize",
                                       json={"content": text}, timeout=20.0)
            n = len(r.json().get("tokens") or [])
        except Exception:
            n = max(1, len(text) // 4)     # a rough estimate is better than none
        if len(text) > 4000:               # only the big, stable ones are cached
            self._tok_cache[key] = n
        return n

    async def _fit(self, msgs, max_tokens):
        """Drop the oldest history until the request fits the window."""
        ctx = await self._n_ctx()
        if not ctx:
            return msgs, max_tokens
        margin = 64                        # chat template, role markers, slack
        counts = [await self._count(m["content"]) for m in msgs]
        need = sum(counts) + margin + max_tokens
        if need <= ctx:
            return msgs, max_tokens
        dropped = 0
        # index 0 is the system prompt and the last is the current message;
        # everything between is history, oldest first
        while need > ctx and len(msgs) > 2:
            need -= counts.pop(1)
            msgs.pop(1)
            dropped += 1
        if dropped:
            print(f"[ctx] dropped {dropped} history turn(s) to fit {ctx} tokens",
                  flush=True)
        if need > ctx:
            fresh = await self._n_ctx(refresh=True)
            if fresh and fresh > ctx:
                print(f"[ctx] window grew {ctx} -> {fresh}", flush=True)
                ctx = fresh
        if need > ctx:
            # nothing left to drop: buy room from the answer instead
            room = ctx - (sum(counts) + margin)
            if room >= 192:
                print(f"[ctx] system+message leaves {room} tokens; "
                      f"capping the answer at that", flush=True)
                return msgs, room
            raise RuntimeError(
                f"the prompt does not fit: system+message is "
                f"{sum(counts)} tokens of a {ctx} window, leaving {room} for the "
                f"answer. Raise --ctx-size on the language model server, or use "
                f"the compact 'codes' style.")
        return msgs, max_tokens

    async def turn(self, message, history=None, max_tokens=512, persona=None,
                   heard=None, identity=None, extra=None):
        """One acting turn -> (parsed dict, latency ms, raw text)."""
        system = self.system
        if extra:
            # An experiment knob: an extra block appended to the standing rules,
            # so prompt variants can be compared without editing the prompt.
            system = system + "\n\n" + str(extra).strip()
        if persona:
            # the character brief goes first; the acting machinery below it is
            # unchanged, so a persona changes who speaks, not what it can do
            system = ("CHARACTER — this is who you are for this whole "
                      "conversation:\n" + persona.strip() + "\n\n" + system)
        # A benchmark item arrives as raw JSON.  Handed that, the local 12B has
        # to parse it, find the line among nine fields and follow the format
        # rules at once, and it answers *about* the situation instead of
        # performing the sentence.  Rewrite it into a plain task first; the JSON
        # never reaches the model.
        item = None
        try:
            import benchmark
            item = benchmark.detect(message)
        except Exception as e:
            print(f"[benchmark] detect failed: {e}", flush=True)
        if item is not None:
            message = benchmark.brief(item)
            system = system + (
                "\n\nTHIS TURN: you have been handed a script to perform. The words "
                "are fixed and are not yours to change — your work is the "
                "directions, the pauses and the bursts you place among them. "
                "Put the given words, annotated, in `script`, and the same words "
                "without annotation in `reply`.")
            print(f"[benchmark] item {item.get('id') or '?'} -> acting brief",
                  flush=True)

        msgs = [{"role": "system", "content": system}]
        # Luna is a reasoning model with a very large context: give it enough of
        # the conversation to actually reason about where the scene has got to.
        # The local model is capped at 8k, so it keeps a shorter tail.
        depth = config.HISTORY_TURNS_LOCAL if self.backend == "local" \
            else config.HISTORY_TURNS_LUNA
        for h in (history or [])[-depth:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        if heard:
            # what the emotion and voice models heard in the user's actual voice
            message = (message + "\n\n[heard in their voice: " + heard +
                       " — respond to how they sound, never mention this note]")
        msgs.append({"role": "user", "content": message})
        msgs, max_tokens = await self._fit(msgs, max_tokens)

        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.95,
            # Gemma-4 is a reasoning model: left on, it spends the whole budget in
            # reasoning_content and returns empty content.  Thinking buys nothing
            # here and costs a second or two straight off time-to-first-audio.
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "acting_turn", "schema": self.schema,
                                "strict": True},
            },
        }
        if self.hosted_model:
            # hosted endpoint: different token parameter, no thinking switch.
            # reasoning_effort=none is the single biggest latency win available
            # here — measured 5.3 s -> 1.5 s on flash-lite, 4.4 s -> 2.4 s on luna
            # — and this turn needs no deliberation, only a character decision.
            body.pop("chat_template_kwargs", None)
            # The cap counts reasoning tokens too, even at effort=none, so a
            # 512 budget silently truncated the JSON mid-string on flash-lite.
            body.pop("max_tokens", None)
            if self.model in getattr(config, "HOSTED_PLAIN_MAX_TOKENS", ()):
                body["max_tokens"] = config.HOSTED_MAX_TOKENS
            else:
                body["max_completion_tokens"] = config.HOSTED_MAX_TOKENS
            body["reasoning_effort"] = config.HOSTED_REASONING
            if self.style == "codes":
                # strict mode there requires additionalProperties:false everywhere
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "acting_turn", "strict": True,
                                    "schema": _harden(self.schema)}}
            else:
                # the prose schema has genuinely optional fields, which strict
                # mode forbids; fall back to free JSON with the shape in the prompt
                body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        t0 = time.time()
        r = await self.client.post(f"{self.base}/v1/chat/completions",
                                   json=body, headers=headers)
        # Some hosted models return an intermittent 400 whose own message says
        # to try again — glm-5.3 did so on one turn in three with a body that
        # was byte-identical to two that succeeded.  One retry, only for a
        # hosted model, and only when the body is not the thing at fault.
        if (r.status_code == 400 and self.hosted_model
                and "client_side_error" in (r.text or "")):
            print(f"[llm] {self.model} returned a retryable 400, trying once "
                  f"more", flush=True)
            await asyncio.sleep(1.0)
            r = await self.client.post(f"{self.base}/v1/chat/completions",
                                       json=body, headers=headers)
        ms = (time.time() - t0) * 1000
        if r.status_code >= 400:
            # A bare "400 Bad Request" says nothing; llama.cpp puts the actual
            # reason in the body, and for this server it is almost always the
            # context window.
            detail = r.text[:400].replace("\n", " ")
            raise RuntimeError(f"{self.backend} {r.status_code}: {detail}")
        j = r.json()
        if not j.get("choices"):
            raise RuntimeError(
                f"{self.backend}: no completion returned "
                f"({json.dumps(j)[:200]})")
        raw = j["choices"][0]["message"].get("content") or ""
        out = None
        try:
            out = _loads_loose(raw)
        except Exception:
            out = None
        if not isinstance(out, dict) or not (out.get("script") or out.get("s")):
            # Falling back to `{"reply": raw}` here put the model's own thinking
            # in the place the performance belongs, which is how this surfaced:
            # a reply that began "The user asks:...".  Ask once more instead.
            # Two retries, not one.  Each attempt fails independently — glm-5.3
            # dumps its reasoning into `content` on roughly one turn in six —
            # so a single retry still leaves about one turn in thirty-six
            # failing outright, which is often enough to meet in a session.
            for attempt in (1, 2):
                print(f"[llm] {self.model} returned no usable script "
                      f"({len(raw)} chars); attempt {attempt} of 2", flush=True)
                r2 = await self.client.post(f"{self.base}/v1/chat/completions",
                                            json=body, headers=headers)
                if r2.status_code < 400 and (r2.json().get("choices") or []):
                    raw2 = r2.json()["choices"][0]["message"].get("content") or ""
                    try:
                        o2 = _loads_loose(raw2)
                        if isinstance(o2, dict) and (o2.get("script")
                                                     or o2.get("s")):
                            out, raw = o2, raw2
                            break
                    except Exception:
                        pass
                    raw = raw2
            ms = (time.time() - t0) * 1000
        if not isinstance(out, dict) or not (out.get("script") or out.get("s")):
            raise RuntimeError(
                f"{self.backend}: the director returned no script. This is "
                f"usually a model putting its reasoning in `content` — try "
                f"another brain, or the same one again.")
        if self.style == "codes" and self.codebook is not None:
            out = self._from_codes(out)
        out = self._clean(out, identity=identity)
        if item is not None:
            out = self._hold_to_script(out, item)
        out = self._breathe(out)
        return out, ms, raw

    @staticmethod
    def _hold_to_script(out, item):
        """A benchmark item is only answered if the words came back unchanged."""
        import benchmark
        sc = out.get("script") or ""
        if benchmark.verbatim_ok(sc, item):
            return out
        print("[benchmark] model drifted off the given script; "
              "using the plain rendering", flush=True)
        out["script"] = benchmark.annotate(item)
        out["reply"] = benchmark.script_text(item)
        return out

    @staticmethod
    def _breathe(out):
        """Give the line its breath back if it came back without any.

        The rules ask for silence inside a sentence and the model still writes
        replies that only stop at full stops, which reads as typed rather than
        spoken.  This never overrides a pause the model chose: it fires only
        when there is none, and only at commas, dashes and the conjunctions
        that begin a new thought.
        """
        if not config.BREATHE_ON:
            return out
        try:
            import benchmark
            sc, n = benchmark.breathe(out.get("script") or "",
                                      limit=config.BREATHE_MAX,
                                      want=config.BREATHE_WANT)
            if n:
                out["script"] = sc
                print(f"[breathe] added {n} pause(s) the model left out",
                      flush=True)
        except Exception as e:
            print(f"[breathe] skipped: {e}", flush=True)
        return out

    def _from_codes(self, o):
        """Expand the compact code answer into the full prose shape."""
        cb = self.codebook
        codes = o.get("d") or ""
        return {
            "voice": cb.reference_for(codes) or {"mode": "none"},
            "voice2": {"mode": "none"},
            "style": cb.style_for(codes),
            "speed": (o.get("sp") if o.get("sp") in config.SPEED_WORDS
                      else "normal"),
            "language": o.get("l") or "English",
            "delivery": cb.expand_delivery(codes),
            "script": cb.expand_script(o.get("s") or ""),
            "codes": codes,
            "blend": cb.blend(codes),
        }

    _TAG = None

    @staticmethod
    def _strip_tags(s):
        """`text` must be the bare spoken words: a stray (cue) or [pause] that leaked
        in from the script would be read out instead of performed."""
        import re
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"\[[^\]]*\]", " ", s)
        return re.sub(r"\s{2,}", " ", s).strip()

    @staticmethod
    def _sanitise_script(s):
        """Two failure modes the prompt alone does not reliably prevent.

        Capitals are spelled out letter by letter by this model, so "AAAGH!"
        becomes "ay-ay-gee-aitch"; and the model sometimes drops a condition key
        like "(ga_pain_scream)" in as if it were a stage direction, which would
        simply be read aloud.
        """
        import re
        # drop cues that are identifiers rather than prose
        s = re.sub(r"\(\s*[a-z0-9]+(?:_[a-z0-9]+)+\s*\)", " ", s, flags=re.I)
        # any run of >=2 capitals becomes lower case ("I" and "AI" style acronyms
        # of one or two letters at a word boundary are left alone)
        s = re.sub(r"\b[A-ZÄÖÜ]{3,}\b", lambda m: m.group(0).lower(), s)
        # The manual is explicit that a pause directly after a BURST truncates
        # it.  This used to match any round bracket, so it also deleted the
        # pause after a delivery direction — which is where a director puts the
        # first silence of a sentence, right behind its cue.  Every reply was
        # quietly losing one, and the ones that survived were the ones written
        # mid-clause.  Only strip it when the bracket really is a burst.
        def _drop_after_burst(m):
            import timed_script as _ts
            body = m.group(1)[1:-1]
            is_burst = bool(re.search(r"[0-9]", body)) or _ts._is_burst_label(body)
            return m.group(1) if is_burst else m.group(0)
        s = re.sub(r"(\([^)]*\))\s*\[[^\]]*pause[^\]]*\]", _drop_after_burst,
                   s, flags=re.I)
        # Square brackets are only for [pause]; a burst written as "[a soft sigh]"
        # would otherwise be stripped as markup and never performed at all.
        s = re.sub(r"\[([^\]]*)\]",
                   lambda m: m.group(0) if "pause" in m.group(1).lower()
                   else "(" + m.group(1) + ")", s)
        # A cue at the very end has nothing to colour and, for a burst, the manual
        # warns the model simply stops there — so a trailing cue is dropped.
        s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
        return re.sub(r"[ \t]{2,}", " ", s).strip()

    @classmethod
    def _clean(cls, out, identity=None):
        delivery = (out.get("delivery") or out.get("general") or "").strip()
        script = cls._sanitise_script((out.get("script") or "").strip())
        # the spoken words are the script minus its direction
        reply = cls._strip_tags(out.get("reply") or script)
        # the model sometimes includes the labels, sometimes not — normalise so the
        # instruction always has the exact two-block shape the model was trained on
        for lab in ("GENERAL:", "DELIVERY:"):
            if delivery.upper().startswith(lab):
                delivery = delivery[len(lab):].strip()
        if script.upper().startswith("SCRIPT:"):
            script = script[len("SCRIPT:"):].strip()
        if not script:
            script = reply

        v = out.get("voice") or {}
        if not isinstance(v, dict):
            v = {"mode": "none"}

        # GENERAL = fixed identity + this turn's delivery + the continuity clause.
        # The identity half is byte-identical on every turn, which is what keeps the
        # speaker from being recast between replies.  A character role is the one
        # deliberate exception: the user asked for a different voice.
        head = (f"the voice of {v.get('character')}." if v.get("mode") == "character"
                and v.get("character")
                else (identity or config.SPEAKER_IDENTITY))
        tail = ("genuine and spontaneous, like a real person in a real moment, not acted. "
                "pristine high-quality studio recording, no background noise.")
        general = " ".join(x for x in (
            head, delivery, config.BASE_REGISTER, config.CONTINUITY, tail) if x)
        # The SAME block with the performance direction removed, which is the
        # "unconditional" half of a guidance pair.  Identity, register, continuity and
        # recording quality are byte-identical in both branches -- only the affect leaves --
        # so the guidance difference is about how the line is delivered and nothing else.
        # Built here rather than by regex downstream because this is the only place that
        # knows which clause was the director's and which are standing.
        general_unc = " ".join(x for x in (
            head, config.BASE_REGISTER, config.CONTINUITY, tail) if x)

        lang = (out.get("language") or "English").strip()
        out["language"] = "German" if lang.lower().startswith("g") or \
            lang.lower().startswith("de") else "English"
        out["reply"] = reply
        out["delivery"] = delivery
        out["general"] = general
        out["general_unc"] = general_unc if delivery else None
        out["script"] = script
        out["instruction"] = f"GENERAL: {general}\nSCRIPT:\n{script}"

        # The mode call.  Unknown words are not an error: they fall back to `auto`, which
        # resolves from the measured table.  A dose never arrives as a number.
        pf = out.get("perform")
        if not isinstance(pf, dict):
            pf = {}
        mode = str(pf.get("mode") or "auto")
        strength = str(pf.get("strength") or "moderate")
        out["perform"] = {
            "mode": mode if mode in levers.ALL_MODE_WORDS else "auto",
            "dimension": (pf.get("dimension") or None),
            "strength": strength if strength in levers.STRENGTHS else "moderate",
        }
        out["voice"] = v
        return out
