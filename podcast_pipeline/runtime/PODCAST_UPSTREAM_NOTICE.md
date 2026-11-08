# Attribution and scope

LAION's Humanness Voice Podcast is a full supervised fine-tune of Humaneness Voice Large 1.7, using the MOSS Local Transformer family implementation and an external MOSS-v2 audio codec. Earlier family provenance and licenses remain in the inherited UPSTREAM files and source copyright headers. The current release's parent, data mixture and checkpoint are documented in TRAINING.md; older inherited notes describe earlier family releases and are not the current model card.

New LAION weights, pipeline code and documentation: CC BY 4.0. Third-party implementation files, Python packages, codecs, ASR models, score models and the optional Voice Converter retain their own licenses and notices. The optional converter is separately downloaded and installed; it is not silently embedded in this repository.

The German example anchors are short excerpts from recordings supplied by the experiment owner. The English anchors are the previously approved short real-podcast reference excerpts. MP3s and exact cached MOSS codes are included for reproducible conditioning; their provenance is in examples/references/PROVENANCE.json. Source recordings retain original terms. The model's CC BY 4.0 notice does not relicense source recordings or grant rights over someone's identity.

Generated example episodes are LAION model outputs, released under CC BY 4.0 with their source-condition records. They demonstrate the earlier script-boundary crop and 400-ms assembly; they are not evidence validating the newly added internal-pause compaction or optional per-turn VC route.
