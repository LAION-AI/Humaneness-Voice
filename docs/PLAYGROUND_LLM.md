# Hosted and local director LLMs

The speech model does not need an LLM for manual TTS. The director is an
optional **text HTTP client**, not part of the speech checkpoint. It interprets
an acting brief or adds directions to fixed spoken text. It does not choose
voices, LoRAs or GPU settings in the current Playground.

## Current hosted setup

The captured defaults in `settings.py` are:

```bash
export SMALL_LUNA_BASE=https://api.hyprlab.io
export SMALL_LUNA_MODEL=gpt-5.6-luna
# Provide SMALL_LUNA_API_KEY using a private shell/secret manager.
```

`LUNA_BASE` must be the base **without** a final `/v1`: the current client
appends `/v1/chat/completions`. It sends a Bearer header, `model`, `messages`,
`reasoning_effort: "none"`, `max_completion_tokens: 2400` and
`response_format: {"type":"json_object"}`. Reward-plan requests use 1400
completion tokens instead. Timeout is 30 s with 10-s connection timeout;
malformed JSON gets one correction retry. This is the implemented protocol,
not an assertion that every similarly named endpoint accepts every field.

Credential lookup prefers `SMALL_LUNA_API_KEY`, then `HYPRLAB_API_KEY`, then
`OPENAI_API_KEY`, then the configured `SMALL_LUNA_KEY_FILE`. The snapshot's
historical default file path is only a location; no credential is included.
Do not put a key in the browser, source control, model card or a public URL.

## Local LLMs: compatible endpoint, same contract

A local llama.cpp, vLLM or other server can act as the director **if it exposes
the required chat-completion interface and can follow the JSON/prompt contract**.
This is a configuration/integration path, not a measured guarantee for every
local model. The separate September root agent already has its own local Gemma
integration; its tool-calling prompt is not the current Luna prompt.

For a local service listening at port 8080, configure:

```bash
export SMALL_LUNA_BASE=http://127.0.0.1:8080
export SMALL_LUNA_MODEL=YOUR_SERVER_MODEL_ID
export SMALL_LUNA_API_KEY=local-placeholder
```

The placeholder is needed because the current client rejects an empty key
before contacting even a key-free local server. Use a real local credential
instead if that server requires one. Bind locally or secure the network;
`127.0.0.1` inside a container is the container itself, not another host.

Before connecting the UI, test whether the endpoint accepts
`reasoning_effort`, `max_completion_tokens` and `response_format`. If it rejects
them, adapt those fields in `Director.direct()` **and** `Director.reward_plan()`
or use a compatibility gateway. Some servers instead expect `max_tokens`;
some models need constrained JSON decoding. The current code does not negotiate
these differences automatically. Check the chosen server/model's current
documentation and its context/output-token limits. No tool-call capability is
needed. A smaller LLM that changes the transcript is not a valid exact-text
replacement merely because it returns parseable JSON.

## What the director must return

```json
{
  "text": "I am glad you came back.",
  "general": "Use the same speaker as the reference audio. Preserve that speaker's voice identity. Gentle, natural and spontaneous delivery.",
  "script": "(warmly relieved) [2.5 seconds duration] I am glad you came back.",
  "caption": "Use the same speaker as the reference audio. Preserve that speaker's voice identity. Gentle, natural and spontaneous delivery.",
  "tags": ["warm", "relieved"],
  "language": "en",
  "delivery_explicit": false
}
```

The exact instructions are [exported in full](LUNA_SYSTEM_PROMPT.md). The client
adds selected format/language, whether reference audio is present, the input
mode and up to 16 recent user/assistant messages. It validates types and spoken
text. In structured chat the stripped SCRIPT is authoritative; in exact-text
mode it must preserve the supplied words exactly after whitespace normalization.
When `delivery_explicit` is explicitly false, the server enforces the gentle/
natural/spontaneous fallback in both GENERAL and CAPTION. Explicit emotion
instructions are not replaced by that fallback.

The separate adaptive-reward request selects 1–4 supported VoiceNet/EIV+
dimensions, objectives/targets/weights from the supplied catalog, never tools
or new arbitrary dimensions. Its exact instructions are in `reward_plan()`;
native bounds and definitions are in `reward_catalog.json`. There are no
“hidden LoRA tools” to expose to a local LLM.

## No LLM or no network

Use `input_mode: "manual"` and a supported prompt format. The backend bypasses
the director; no LLM credential is needed. All audio weights, codec assets and
any enabled scorers must still be cached locally to run offline. Pointing the
director at localhost alone does not make Hugging Face downloads offline.
Disable Luna automatic reward planning or select dimensions manually when no
director is available.
