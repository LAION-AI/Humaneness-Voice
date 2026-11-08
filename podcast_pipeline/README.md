# Podcast pipeline

See [the complete recipe, model links, reproduction commands and speed measurements](../docs/PODCAST_GENERATION_PIPELINE.md).

Default: listener-approved sparse spontaneous, relaxed and gentle inline prompting, full Podcast checkpoint, no voice conversion, complete-turn Best-of-4 with 50% native-normalized Ears genuineness and 50% ECAPA reference cosine. WER is a final diagnostic only.

Runtime files are extracted from the tested Humaneness inference implementation. `runtime/published_audio.py` is the Podcast release's word-clock assembly code; its upstream license and attribution are retained next to it. Other third-party notices remain in source headers. Reference recordings are downloaded separately and retain their original terms. No API keys, recordings from Robby's microphone, service environments, model weights or generated episodes are committed.

[Exact prompting templates and LLM-writing instructions](PROMPTING.md). Use `produce.py` for a complete generation → audit → MP3/HTML command. `compare_n.py` shares loaded weights across the fresh N=2 / N=1 comparison.

For a shared-memory fallback, `low_memory_compare.py` generates N=2 and N=1 acoustic codes with the policy alone in VRAM, then releases the policy before decoding and ranking. `compare_report.py` builds a six-player comparison from audited outputs. See the manual for timing caveats.
