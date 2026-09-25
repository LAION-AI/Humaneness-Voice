"""Check that the documentation still describes the code.

    python setup/check_docs.py

Three kinds of drift are caught here, all of which have actually happened:

* a generated page that was not regenerated after a prompt change
  (`SYSTEM_PROMPTS.md` was two days stale and still described an older director);
* a documented default that no longer matches `config.py`;
* a link to a page that does not exist.

Exit code 1 if anything is off, so it can go in front of a commit.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
DOCS = os.path.join(HERE, "docs")

# what DEFAULTS.md and the other pages promise, and where it really lives
CLAIMS = [
    ("SIDON_ON", False), ("SIDON_BASE", "http://127.0.0.1:8793"),
    ("BON_GUIDANCE", 3.0), ("SPEAK_BEST_OF", 10), ("SPEAK_GUIDANCE", 3.0),
    ("BURST_LAM_BUDGET", 2.0), ("BURST_LAM_MAX", 1.25),
    ("BREATHE_ON", True), ("BREATHE_WANT", 3), ("BREATHE_MAX", 3),
    ("ENGLISH_CUES", True), ("BON_BURST_READY", True),
    ("BURST_QUANT", "4bit"), ("ALIGN_ON", True), ("SKILLS_ON", True),
    ("PROFILE_LORA_LAM", 1.0), ("SFT3_EMOTION_LAM", 1.0),
]

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main():
    import config

    print("defaults, against config.py")
    for name, want in CLAIMS:
        got = getattr(config, name, "<missing>")
        check(got == want, f"{name} = {want!r} (is {got!r})")

    print("\ngenerated pages, against the code")
    import llm_agent
    p = os.path.join(DOCS, "SYSTEM_PROMPTS.md")
    if os.path.exists(p):
        cur = open(p, encoding="utf-8").read()
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as t:
            tmp = t.name
        subprocess.run([sys.executable,
                        os.path.join(HERE, "setup", "build_system_prompts.py"),
                        "--out", tmp], check=True,
                       stdout=subprocess.DEVNULL)
        fresh = open(tmp, encoding="utf-8").read()
        os.unlink(tmp)
        check(cur.strip() == fresh.strip(),
              "SYSTEM_PROMPTS.md is what build_system_prompts.py produces")
    # every headline rule of the prompt should be findable in the page
    for block in ("THE CLOCK IS PART OF THE ACTING",
                  "HESITATION SOUNDS ARE ALLOWED",
                  "MOST OF YOUR PAUSES BELONG BETWEEN TWO WORDS",
                  "EVERY BRACKET IS WRITTEN IN ENGLISH",
                  "EVERY SENTENCE CARRIES A DIRECTION"):
        check(block in llm_agent.SYSTEM, f"prompt still contains {block!r}")

    print("\ninternal links")
    seen = set()
    for f in ["README.md"] + [os.path.join("docs", x)
                              for x in sorted(os.listdir(DOCS))]:
        path = os.path.join(HERE, f)
        if not path.endswith(".md"):
            continue
        for m in re.finditer(r"\]\((docs/)?([A-Za-z0-9_./-]+\.md)\)",
                             open(path, encoding="utf-8").read()):
            tgt = m.group(2)
            if tgt in seen:
                continue
            seen.add(tgt)
            ok = os.path.exists(os.path.join(DOCS, tgt)) or \
                os.path.exists(os.path.join(HERE, tgt))
            if not ok:
                check(False, f"{f} links to missing {tgt}")
    check(True, f"{len(seen)} link targets, all present"
          if not any("links to missing" in x for x in fails) else "links")

    print("\nthe format's own arithmetic")
    import timed_script
    line = ("(clearly amused) [2.0 seconds duration] A test line here. "
            "[0.6 seconds pause] And another one after it.")
    tagged, frames, plain = timed_script.render(line)
    total = sum(float(x) for x in re.findall(
        r"\[([0-9.]+) seconds (?:pause|duration)\]", tagged))
    check(abs(total * config.FRAME_RATE - frames) < 1,
          "rendered seconds sum to the token count")
    check("[" not in plain and "(" not in plain,
          "the spoken text carries no markup")
    check(timed_script.STANDING in timed_script.general_line("x", 1.0, "EN"),
          "every GENERAL carries the standing block")

    print("\nhugging face urls in DEPENDENCIES.md")
    dep = os.path.join(DOCS, "DEPENDENCIES.md")
    if os.path.exists(dep) and "--urls" in sys.argv:
        from huggingface_hub import HfApi
        api = HfApi()
        # The namespace is part of the URL and has to be read out of it: the
        # earlier version consumed only a `datasets/` prefix, so a Space URL
        # parsed as the repository `spaces/<owner>` and was then looked up in
        # the model namespace, where it cannot exist.  That reported a live
        # Space as a dead link -- the check meant to catch documentation drift
        # produced some.  Try the declared namespace first and the others
        # after, so a link is dead only when NO namespace has it.
        KIND = {"datasets/": "dataset", "spaces/": "space", None: "model"}
        FN = {"model": api.model_info, "dataset": api.dataset_info,
              "space": api.space_info}
        found = {}
        for m in re.finditer(
                r"https://huggingface\.co/(datasets/|spaces/)?([\w.-]+/[\w.-]+)",
                open(dep, encoding="utf-8").read()):
            found.setdefault(m.group(2), set()).add(KIND[m.group(1)])
        bad = []
        for repo, kinds in sorted(found.items()):
            order = sorted(kinds) + [k for k in FN if k not in kinds]
            for k in order:
                try:
                    FN[k](repo)
                    break
                except Exception:
                    continue
            else:
                bad.append(repo)
        check(not bad, f"{len(found)} repositories resolve"
              + (f" (missing: {bad})" if bad else ""))
    else:
        print("  skipped (pass --urls to hit the network)")

    print()
    if fails:
        print(f"{len(fails)} problem(s).")
        return 1
    print("everything checks out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
