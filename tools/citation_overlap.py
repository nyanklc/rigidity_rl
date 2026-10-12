"""Close-paraphrase screen: word n-grams a thesis chapter shares with a source text.

Strips LaTeX commands and inline math, lowercases, and reports every run of shared
n-grams with its length. A run is a lead to read by eye, not a verdict: generic technical
phrases ("rank of the rigidity matrix") match too. Extract sources with `pdftotext -layout`,
since `-raw` glues words on some IEEE PDFs.

    pdftotext -layout paper.pdf paper.txt
    uv run tools/citation_overlap.py chapter.tex paper.txt [more.txt ...] [--n 7]
"""
import argparse
import os
import re


def words(text):
    text = re.sub(r"\\(cite|ref|eqref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.findall(r"[a-z]+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--show", type=int, default=8, help="longest runs printed per source")
    args = ap.parse_args()
    N = args.n

    tex = re.sub(r"(?m)^\s*%.*$", "", open(args.tex, encoding="utf8").read())
    tw = words(tex)
    tex_grams = [tuple(tw[i:i + N]) for i in range(len(tw) - N + 1)]
    for src in args.sources:
        sw = words(open(src, errors="ignore").read())
        shared = set(tex_grams) & {tuple(sw[i:i + N]) for i in range(len(sw) - N + 1)}
        runs = []
        for i, g in enumerate(tex_grams):
            if g not in shared:
                continue
            if runs and i <= runs[-1][1] + 1:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        runs = sorted(((a, b + N) for a, b in runs), key=lambda r: r[0] - r[1])
        print(f"== {os.path.basename(src)}: {len(shared)} shared {N}-grams, {len(runs)} runs")
        for a, b in runs[:args.show]:
            print(f"   [{b - a}w] " + " ".join(tw[a:b]))


if __name__ == "__main__":
    main()
