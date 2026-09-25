# The exact prompts used in the arena run

Reproduced verbatim so the run can be repeated or argued with. See
[`ARENA.md`](ARENA.md) for what they produced.

## The five seeded blocks (generation 1)

Each is appended to the standing system prompt, after the persona and the
acting rules. `control` is the empty string: the prompt as shipped.

### `control`

*(no addition — the standing prompt alone)*

### `imperfection`

```
ONE MORE THING FOR THIS REPLY. Real speech is never clean. Build in the small failures a person makes and does not notice: a word started and restarted, a breath taken in the wrong place, a thought that arrives before the sentence is ready for it. Mark them with directions and pauses where they fall. A take that is slightly untidy in the way a person is untidy beats a take that is smooth.
```

### `breath`

```
ONE MORE THING FOR THIS REPLY. Put more silence in it than feels correct on the page. Between two and four pauses, most of them INSIDE sentences rather than between them, placed where the speaker is thinking, hesitating, choosing a word, or deciding whether to go on. Vary their length: 0.2 for a breath, 0.5 for a beat, 0.9 when the thought genuinely stalls.
```

### `body`

```
ONE MORE THING FOR THIS REPLY. This voice has a body. Use vocal bursts for what the body does while the mind talks — the intake before bad news, the laugh that escapes before it is approved, the sigh that lands after the sentence rather than before it. At least one burst, given its own bracket and its own length, placed where the body would act rather than where it would be tidy.
```

### `subtext`

```
ONE MORE THING FOR THIS REPLY. Play what is underneath, not what is on top. The feeling should be visibly held rather than performed: contained, leaking at the edges of phrases, showing in the timing and the breath before it shows in the volume. Trust the listener to hear a small signal. Under-play by one notch on the adverb scale rather than over-play.
```

## The breeding prompt

Sent to `gemini-3.8-flash` between generations, with the six
best-scoring blocks so far and their rubric scores substituted in.

```
You are tuning the system prompt of a voice director. The director
writes a timed script — delivery directions in round brackets, vocal bursts with
a length, pauses in square brackets — which a text-to-speech model then performs.

An extra block of instruction is appended to the director's standing rules. We
are searching for the block that produces the best-sounding performance.

Each block below was tried and the resulting audio was rated 0-5 by a listener
on three rubrics: PLEASANT (how pleasant it sounds), FIT (how well it matches
the task) and NATURAL (how much it sounds like a real, spontaneous moment,
including imperfections and micro-expressions).

%s

Write %d NEW blocks. Each should be a different bet, not a rewording of the
winners. You may combine what worked, push a winning idea further, or try
something none of them tried. Keep each under 90 words, addressed to the
director in the second person, and phrased as instructions for THIS reply.
Do not mention scores, rubrics, experiments or this message.

Return JSON: {"blocks": [{"name": "<two words, lowercase, hyphenated>",
"text": "<the block>"}, ...]}
```

## The judge's prompt

Sent with each clip as `input_audio`. The task fields come from the
benchmark item.

```
You are listening to a single take from a voice-acting
benchmark. Rate it on three rubrics, each 0 to 5, each with one sentence of
justification.

THE TASK THE ACTOR WAS GIVEN
Situation: %s
Direction: %s
Emotion to convey: %s (%s intensity)
The words, which are fixed: "%s"

THE RUBRICS
- pleasant: how pleasant this is to listen to. 0 is grating, distorted or
  painful; 5 is a voice you would happily hear for an hour.
- fit: how well the performance matches the task above — the emotion, its
  intensity, the situation and the direction. 0 is unrelated or contradictory;
  5 is exactly what was asked for.
- natural: how much this sounds like a real person in a real, spontaneous
  moment, with the imperfections and micro-expressions that come with it —
  breath, hesitation, a word caught, timing that is not metronomic. 0 is
  obviously synthetic or read aloud; 5 is indistinguishable from a candid
  recording.

Judge only what you hear. Ignore recording quality differences that are not the
performance. Be willing to use the whole range: most takes are not 4s.

Return JSON: {"pleasant": {"score": n, "why": "..."},
"fit": {"score": n, "why": "..."}, "natural": {"score": n, "why": "..."}}
```
