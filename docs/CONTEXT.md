# The context window is the binding constraint on the local model

The local brain is `gemma-4-12B-it-qat` served by llama.cpp. Its system prompt is
not small: the acting rules, the tool schemas, the voice catalogue and the burst
block together come to **6,917 tokens**, and a character persona adds roughly
**1,100** more. Against the 8,192-token window the server shipped with, that left
about 90 tokens for the answer — and any turn that needed more came back as a
bare

```
⚠ llm: Client error '400 Bad Request' for url 'http://127.0.0.1:8790/v1/chat/completions'
```

with nothing to say which of a dozen possible causes it was.

Three things were wrong, and all three are fixed.

## 1. The error said nothing

`raise_for_status()` throws away the response body, which is exactly where
llama.cpp puts the reason. The failure now carries it:

```
local 400: the request exceeds the available context size…
```

## 2. Nothing checked before sending

`LLMAgent._fit()` now measures the request with the model's own tokeniser
(`/tokenize`), reads the real window from `/props`, and drops the oldest history
turns until it fits. If there is no history left to drop it buys room from the
answer instead, and only when that would leave under 192 tokens does it refuse —
with the numbers in the message:

```
the prompt does not fit: system+message is 8382 tokens of a 8192 window,
leaving -254 for the answer. Raise --ctx-size on the language model server,
or use the compact 'codes' style.
```

The window is re-probed once before refusing, because the language model server
can be restarted with a bigger one while this process keeps running — which is
precisely what happened while this was being fixed, and a cached 8192 made a
working 16384 server look broken.

## 3. The window was too small for the prose style at all

Measured, with a persona loaded and no history:

| style | system + message | fits in 8192? |
|---|--:|---|
| `codes` (default) | ~2,400 | yes, comfortably |
| `prose` | 8,038 | no — 90 tokens left for the answer |
| `prose` + benchmark item | 8,382 | no — over by 254 |

So `--ctx-size` is now `${MOSS_LLM_CTX:-16384}` in `run.sh`. Doubling the KV
cache cost nothing that mattered: GPU 0 went from 20,611 MiB to 19,779 MiB in
use of 24,576, because the model itself dominates. All three paths — prose with
an item, prose without, codes with an item — pass afterwards.

**If you run this on a smaller card** and cannot afford 16k, use the `codes`
style: it carries the same acting rules in a compact code legend and fits an 8k
window with room to spare. The guard will keep either style from failing
silently.
