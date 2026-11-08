# Prompting guide: spontaneous, relaxed and gentle

This is the listener-approved default as of 8 November 2026 (`--prompt-style spontaneous`). It keeps the original restrained delivery and adds sparse inline cues for spontaneous, easy, gently spoken conversation. The listener preferred these samples; that does not establish a universal optimum for all voices or scripts. The older `legacy` turn-level style and exaggerated `sentence` experiment remain explicit reproduction options.

## Use an LLM as the script and direction writer

Writing every literal turn and every carefully placed acting cue by hand is unnecessarily laborious. We recommend asking an API-based or local language model to produce the JSON script. The LLM writes the dialogue and sparse directions; this Podcast model speaks them. No particular LLM vendor, API key, or network service is required by inference. Review factual content before synthesis.

Give the writing model this instruction:

> Write a natural, thoughtful spoken conversation in the requested language. Use the requested speakers, with stable IDs. Separate literal spoken text from performance cues. Keep each turn short enough to fit comfortably below 25.6 seconds; split long explanations into turns. Make the exchange spontaneous, relaxed, gentle and genuine, with modest contextual emotion rather than theatrical intensity. Use one short cue at the start of a turn and at most one additional cue at a meaningful existing sentence boundary. Do not label every sentence with a strong emotion. Do not request forced laughter, excessive breathiness, exaggerated slowness, or “very pleasant/nice to listen to.” Do not insert speaker labels, directions or duration requests into the spoken text. Output valid JSON with `language` and an ordered `turns` array. Each turn has `index`, `speaker`, `text`, `direction`, and optionally one or two `delivery_chunks`, each containing `text` and `cue`. Joining chunk texts with one space must exactly reproduce the turn's literal `text`. Do not change a supplied transcript. Use no generated audio as a reference; references are selected separately.

Include the topic, factual constraints, speaker roles, intended audience and desired approximate episode length in the writing request. For the evaluated example, two hosts discuss meaning, Viktor Frankl, everyday care and connection. A calm question, gentle curiosity, understated warmth or mild amusement usually suffice. Strong emotions should arise from actual content and an explicit request.

## Exact approved GENERAL

```text
A warm, emotionally thoughtful two-host podcast conversation. Very authentic, very genuine. Natural, intimate, spontaneous spoken delivery with clear words, subtle emotion, relaxed conversational pacing and brief meaningful pauses. Be sincere, without an exaggerated announcer voice or forced laughter. Perform only the current speaker turn and speak exactly the SCRIPT. Keep the same speaker identity as the reference audio.
```

GENERAL remains constant across the tested episodes. It does not include a voice gender, age, timbre redesign, or “very pleasant to listen to.” Identity comes from the fixed reference audio. For a turn within a podcast, use this exact text even though the individual call contains only one speaker.

## Single-speaker call template

```text
GENERAL: [exact GENERAL above]
SCRIPT:
(warm, gently curious; relaxed, spontaneous, gently spoken) What if meaning is something we practise?
(gently, as a spontaneous thought) Maybe it begins with noticing who needs us today.
```

The processor receives the separate literal text `What if meaning is something we practise? Maybe it begins with noticing who needs us today.` and **only this speaker's** reference codes. Inline directions belong in `instruction`, not in the literal transcript. Use `tokens=None` and no inline duration brackets: the 320-frame generation ceiling is a limit, not a requested duration.

Prefer cues such as:

- `easy, spontaneous, gently spoken`
- `relaxed, unforced, like talking to a friend`
- `soft natural delivery, thinking aloud`
- `casual and sincere, unhurried`
- `gentle conversational flow, in the moment`

Do not stack all of them onto one sentence. The approved example files rotate a single short cue and add one extra cue inside alternating turns. Most emotion remains subtle. The second cue is optional.

## JSON accepted by the optimized pipeline

```json
{
  "language": "en",
  "turns": [
    {
      "index": 0,
      "speaker": "S1",
      "text": "What if meaning is something we practise? Maybe it begins with noticing who needs us today.",
      "direction": "warm, gently curious",
      "delivery_chunks": [
        {"text": "What if meaning is something we practise?", "cue": "warm, gently curious; relaxed, spontaneous, gently spoken"},
        {"text": "Maybe it begins with noticing who needs us today.", "cue": "gently, as a spontaneous thought"}
      ]
    },
    {
      "index": 1,
      "speaker": "S2",
      "text": "Yes. Perhaps it begins with a small gesture that makes a real difference to someone.",
      "direction": "warm and reflective"
    }
  ]
}
```

Use one consistent language per file. Supported codes are `en`, `de`, `nl`, `fr` and `es`; the bundled spontaneous scripts cover all five. Set `language` to `de` and write all turns in German for a German episode. A simpler turn with only `speaker`, `text` and `direction` is valid: the default adds one sparse spontaneous cue without changing the literal words. `index` defaults to its position. Pass one `--script` argument per language in a job. Explicit routing preserves the requested language instead of treating non-German text as English. The CLI validates the exact chunk-to-text reconstruction and rejects ambiguous duplicate language/index pairs.

Pass a separate reference mapping with `--references` when changing voices:

```json
{
  "S1": {"path": "audio/voice-one.wav"},
  "S2": {"path": "audio/voice-two.wav"}
}
```

Paths are relative to this reference JSON. Audio SHA-256 values are recorded automatically when absent, and verified when supplied. The bundled mapping additionally provides pinned cached MOSS codes. Remove or regenerate `codes` and `code_key` when replacing audio: otherwise the old codes remain the actual conditioning input. Use clean single-speaker references around 5–15 seconds and reuse them for the entire episode.

## Multi-speaker episodes versus native two-speaker blocks

The recommended optimized route generates each speaker turn separately, always with its correct reference, and then assembles the turns chronologically. A GPU batch can contain different speakers and texts, but their generated histories and rewards stay separate. This is a multi-speaker **episode**, not two speakers mixed into one model call. Any supported speaker ID in the reference mapping can route an independently generated turn; only the two-voice example has been evaluated here.

The preserved native pair route (`infer_legacy.py --mode pairs`) instead uses the trained S1/S2 reference wrapper and `(speaker=S1, ...)` / `(speaker=S2, ...)` instructions within one call. Labels must not appear in the literal transcript. This route has different content checks and rewards, and is not covered by the new optimized Best-of-N timing table. See the model repository's multi-speaker guide before using it. Do not pass two references to the optimized single-turn call and assume it will switch identities correctly.

## Settings and selection

BF16 policy, temperature **1.0**, CFG **2.5**, top-p **0.95**, top-k **50**, free timing, up to **320 × 80 ms = 25.6 s per turn**. No voice conversion, DPO or extra LoRA. Default N=4; use `--n 2` or `--n 1` to trade selection opportunity for less compute.

For N>1, choose the maximum of `0.5*clip(Ears native genuineness/6,0,1) + 0.5*clip(ECAPA reference cosine,0,1)` per complete raw turn. This is fixed normalization, not candidate-wise min/max. WER and Content Enjoyment are not selection terms. N=1 retains its sole take and skips online scoring. All modes receive the same independent final audit.

Quality checks are essential: a highly genuine, similar-sounding take can still omit words. The experiment keeps strict ranking results and flags such failures. If a complete listening copy is repaired using another candidate, document that additional content eligibility check explicitly; never present it as unconstrained reward argmax or as an unchanged Single Take.
