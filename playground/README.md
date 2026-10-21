# October 2026 Playground source snapshot

Start with the repository [README](../README.md), [models](../docs/PLAYGROUND_MODELS.md),
[installation](../docs/PLAYGROUND_INSTALL.md) and [exact Luna prompt](../docs/LUNA_SYSTEM_PROMPT.md).

`runtime/` contains the current Small/Large UI, backend, RL loader, ranking,
streaming/conversion/search code and research harnesses. The adjacent Python
files are the parent compatibility modules that this integration imports.
They are separate from the older portable root application; this deliberately
preserves the working integration rather than silently replacing root files.

The October 21 stop correction adds opt-in `cancel_on_disconnect` to `/api/turn`.
Robby enables it for its own streamed requests. Closing that request stops its
worker at the next cooperative audio callback; the GPU busy flag is released
only after the worker actually unwinds. Existing Playground requests default to
`false` and retain disconnect-survival behavior. There is no global cancellation
endpoint that could stop another user's job. See `runtime/turn_cancellation.py`,
its CPU-only unit test, and [Robby's current test report](../robby/static/stop-fix-update.html).

The published parent `bestofn.py` adds the path-only `HUMANENESS_SCORER_DIR`
override; its scoring math is unchanged. The lightweight RL schema test uses
no-reference mode so it needs no private profile files. These publication
portability adjustments have not been deployed to the working-host server.

The captured code retains absolute working-host paths, including scorer
packages, catalogs, scripts and a historical credential-file *location*.
**The credential file/value, references, generated user audio and model caches
are not included.** Use environment configuration and the upstream assets
described in the installation guide. The full demo is not self-contained just
because its Python source is present. Do not run the historical service/waiter/
publication/restart scripts on another host without reviewing their targets.

`DEVELOPMENT_LOG.md` preserves the dated implementation notes. Headings such
as “current defaults” inside it mean current **at that section's date**, not
today's settings. Use `manifests/playground-2026-10-16.json` for current defaults.
`experiments/` contains disclosed designs/summaries without source audio;
`evidence/` contains selected actual smoke/browser records. Audio URL/session
IDs inside smoke records are ephemeral, not durable recordings or credentials.
`environment-2026-10-16.json` records observed package versions, not a guaranteed
fresh dependency solve. `source-manifest.json` checks file integrity.
The copied research sources also retain historical whitespace; a single
trailing blank line was removed in the published test file for Git checks.

To regenerate metadata and the verbatim prompt export after a deliberate
snapshot update, run `python setup/build_playground_release.py` from repository
root. It reads public Hub metadata, not weights or credentials. Public inventories
can advance; their `inspected_revision` is not a substitute for runtime pins.
Run `python setup/check_playground_docs.py` to verify local links, defaults,
the verbatim Luna prompt, source hashes, Python syntax and publication boundaries.
