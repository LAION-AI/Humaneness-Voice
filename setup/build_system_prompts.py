"""Regenerate docs/SYSTEM_PROMPTS.md from the code that sends the prompts.

    python setup/build_system_prompts.py [--out docs/SYSTEM_PROMPTS.md]

The point of the file is that it cannot drift from what the server actually
sends, which only holds if this is run after the prompt changes. It was not, for
two days, and the page silently described an older director.

Note that `docs/PROMPTING.md` is NOT generated — it is prose about the format
and the measurements behind it, written by hand. An earlier generator wrote
into that file and destroyed those sections; this one only touches the file
named below.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "docs",
                                                  "SYSTEM_PROMPTS.md"))
    a = ap.parse_args()

    import llm_agent
    import personas

    P = ["# System prompts, verbatim", "",
         "Generated from `llm_agent.py` and `personas.py` by",
         "`setup/build_system_prompts.py`, so this file cannot drift from what",
         "the server actually sends. Re-run it after any prompt change.", ""]

    P += ["## Director system prompt (prose mode, the default)", "",
          "```", llm_agent.SYSTEM.strip(), "```", ""]

    code = getattr(llm_agent, "CODE_SYSTEM", "")
    if code:
        P += ["## Director system prompt (code mode)", "",
              "The compact alternative: the same acting rules carried in a code",
              "legend instead of prose. About 2,400 tokens against 6,900, which",
              "is what makes an 8k context window workable — see",
              "[`CONTEXT.md`](CONTEXT.md).", "",
              "```", code.strip(), "```", ""]

    mut = getattr(llm_agent, "MUTATOR", None)
    if mut:
        P += ["## Prompt-variant breeding", "", "```", mut.strip(), "```", ""]

    P += ["## Character briefs", "",
          "Prepended to the director prompt when a persona is chosen. Each one",
          "changes who speaks, not what the director can do.", ""]
    for pid, meta in sorted(personas.BY_ID.items()):
        brief = personas.brief_for(pid) or ""
        name = (meta or {}).get("name") or pid
        tag = (meta or {}).get("tag") or ""
        P += [f"### `{pid}` — {name}" + (f" · *{tag}*" if tag else ""), "",
              "```", str(brief).strip(), "```", ""]

    text = "\n".join(P).rstrip() + "\n"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {a.out}, {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
