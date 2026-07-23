#!/usr/bin/env python3
"""
Task #99 — Paper.md → Paper.pdf conversion via xelatex.

Reads papers/paper.md, splits into 8 logical sections (Title + Abstract,
7 sections + Appendix + References), generates a LaTeX wrapper with
required packages (amsmath, amsthm, booktabs, hyperref, geometry),
preserves inline LaTeX math ($..$, $$..$$) and code blocks verbatim,
and compiles via xelatex twice (for cross-references / TOC).

Usage:
    python3 scripts/task99_md_to_pdf.py [--out papers/paper.pdf]

No network. No fallback (R2). Compilation failure => raise (R2).
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

# LaTeX special chars that need escaping in normal text.
# Skip escaping inside $...$ math regions.
_LATEX_SPECIALS = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$"}


def escape_latex_outside_math(text: str) -> str:
    """Escape LaTeX special chars but skip inside math regions.
    Tracks three states (text / inline-math / display-math) so that
    ambiguous constructs like `$\\kappa$$\\to$0` (two adjacent inline
    math regions) don't get mistaken for a single `$$...$$` display.
    """
    out_parts: list[str] = []
    i = 0
    n = len(text)
    # Tri-state: 0=text, 1=inline-math, 2=display-math.
    state = 0

    def in_math() -> bool:
        return state != 0

    while i < n:
        # Display math delimiters \[ \] — always unambiguous.
        if i + 1 < n and text[i] == "\\" and text[i + 1] == "[":
            state = 2
            out_parts.append("\\[")
            i += 2
            continue
        if i + 1 < n and text[i] == "\\" and text[i + 1] == "]":
            state = 0
            out_parts.append("\\]")
            i += 2
            continue
        # Inline math delimiters \( \)
        if i + 1 < n and text[i] == "\\" and text[i + 1] == "(":
            state = 1
            out_parts.append("\\(")
            i += 2
            continue
        if i + 1 < n and text[i] == "\\" and text[i + 1] == ")":
            state = 0
            out_parts.append("\\)")
            i += 2
            continue
        # Dollar math: ambiguous between single $ (inline) and $$ (display).
        if text[i] == "$":
            is_double = i + 1 < n and text[i + 1] == "$"
            if is_double:
                if state == 0:
                    # $$ in text: open display math.
                    state = 2
                    out_parts.append("$$")
                    i += 2
                    continue
                if state == 2:
                    # $$ in display: close display math.
                    state = 0
                    out_parts.append("$$")
                    i += 2
                    continue
                # state == 1 (inline): treat $$ as two single $ tokens.
                # First $ closes inline (state 1 -> 0).
                state = 0
                out_parts.append("$")
                i += 1
                # Second $ now consumed below as a fresh single $ in text.
                continue
            else:
                # Single $ toggles inline.
                if state == 0:
                    state = 1
                elif state == 1:
                    state = 0
                else:  # display -> inline
                    state = 1
                out_parts.append("$")
                i += 1
                continue
        if in_math():
            out_parts.append(text[i])
            i += 1
            continue
        ch = text[i]
        if ch in _LATEX_SPECIALS:
            out_parts.append(_LATEX_SPECIALS[ch])
        elif ch == "^":
            out_parts.append(r"\^{}")
        elif ch == "~" and (i == 0 or text[i - 1] != "\\"):
            out_parts.append(r"\textasciitilde{}")
        else:
            out_parts.append(ch)
        i += 1
    return "".join(out_parts)

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PAPER_MD = REPO / "papers" / "paper.md"
PAPER_TEX = REPO / "papers" / "paper.tex"
PAPER_PDF = REPO / "papers" / "paper.pdf"


def parse_md_to_latex(md_text: str) -> str:
    """Convert paper.md (after title + abstract) to LaTeX body.
    Preserve $...$, $$...$$, fenced code blocks verbatim.
    Map #/##/### to \\section/\\subsection/\\subsubsection.

    Preprocess: replace $$...\\tag{N}$$ with \\[...\\quad\\text{(N)}\\] since
    \\tag requires equation environment not plain $$...$$.

    Strategy: skip everything up to and including the first '## ' heading's
    preceding separator ('---'), and start the body at the first '## ' line.
    """
    # Preprocess: convert \tag{N} usage to inline text label.
    # IMPORTANT: do NOT use re.DOTALL — that makes . match newlines, which would
    # cause a single $$...\\tag{N}$$ to span arbitrary distance and eat unrelated text.
    # Match both $$...\\tag{N}$$ and the odd $$...\\tag{N}$ (single $ open, $$ close).
    md_text = re.sub(
        r"\$\$?([^\$\n]+?)\\tag\{([^}]+)\}\$\$",
        lambda m: r"\[" + m.group(1).strip() + r"\quad\text{(" + m.group(2) + r")}\]",
        md_text,
    )
    lines = md_text.split("\n")
    # Find first `## ` line (start of section 1)
    body_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            body_start_idx = i
            break
    if body_start_idx is None:
        raise ValueError("No '## ' heading found in paper.md")
    body_lines = lines[body_start_idx:]
    out: list[str] = []
    in_code = False
    code_lang = ""
    code_buf: list[str] = []

    def flush_code() -> None:
        nonlocal code_buf, code_lang
        if code_buf:
            # Map known languages; fall back to text for unknown
            lang_map = {
                "latex": "tex", "tex": "tex", "python": "python",
                "python3": "python", "bash": "bash", "sh": "bash",
                "": "text",
            }
            lang = lang_map.get(code_lang.lower(), "text")
            out.append("\\begin{lstlisting}[language=" + lang + "]")
            out.extend(code_buf)
            out.append("\\end{lstlisting}")
            code_buf = []
            code_lang = ""

    for line in body_lines:
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_lang = line.strip().lstrip("`").strip()
            continue
        if in_code:
            code_buf.append(line)
            continue

        if line.startswith("## "):
            flush_code()
            out.append("")
            out.append("\\section{" + line[3:].strip() + "}")
            continue
        if line.startswith("### "):
            out.append("")
            out.append("\\subsection{" + line[4:].strip() + "}")
            continue
        if line.startswith("#### "):
            out.append("")
            out.append("\\subsubsection{" + line[5:].strip() + "}")
            continue

        # Horizontal rules
        if re.match(r"^-{3,}$", line.strip()):
            out.append("\\noindent\\rule{\\textwidth}{0.4pt}")
            continue

        out.append(line)

    flush_code()
    return "\n".join(out)


def extract_title_abstract(md_text: str) -> tuple[str, str]:
    """Extract (title, abstract) from paper.md. Abstract block starts with '> **Abstract.**'."""
    title = ""
    abstract_lines: list[str] = []
    in_abstract = False
    for line in md_text.split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.strip().startswith("> **Abstract.**"):
            in_abstract = True
            # Strip the prefix
            abstract_lines.append(line.strip().lstrip("> ").replace("**Abstract.**", "").strip())
            continue
        if in_abstract:
            if line.strip().startswith("> "):
                abstract_lines.append(line.strip().lstrip("> ").strip())
            elif line.strip() == "":
                abstract_lines.append("")
            else:
                break
    abstract = " ".join(l for l in abstract_lines if l).strip()
    return title, abstract


def build_latex(title: str, abstract: str, body: str) -> str:
    """Wrap title + abstract + body in a complete LaTeX article.
    All three are pre-escaped via escape_latex_outside_math()."""
    title_tex = escape_latex_outside_math(title)
    abstract_tex = escape_latex_outside_math(abstract)
    body_tex = escape_latex_outside_math(body)

    return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
\usepackage{listings}
\usepackage{xcolor}
\lstset{basicstyle=\ttfamily\small, breaklines=true, frame=single}

% Define commonly-used math operators not built into amssymb.
\DeclareMathOperator{\arctanh}{arctanh}
\DeclareMathOperator{\argmin}{arg\,min}
\DeclareMathOperator{\argmax}{arg\,max}

\title{""" + title_tex + r"""}
\author{Reproduction Paper --- 2026}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
""" + abstract_tex + r"""
\end{abstract}

""" + body_tex + r"""

\end{document}
"""


def compile_xelatex(tex_path: Path, pdf_path: Path) -> None:
    """Run xelatex twice for cross-refs."""
    for run in (1, 2):
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory", str(tex_path.parent),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(tex_path.parent),
        )
        if result.returncode != 0:
            print("=== xelatex STDOUT (last 60 lines) ===", file=sys.stderr)
            print("\n".join(result.stdout.splitlines()[-60:]), file=sys.stderr)
            print("=== xelatex STDERR (last 20 lines) ===", file=sys.stderr)
            print("\n".join(result.stderr.splitlines()[-20:]), file=sys.stderr)
            raise RuntimeError(f"xelatex run {run} failed with returncode={result.returncode}")
    # Verify PDF was created
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not produced at {pdf_path}")
    print(f"OK: paper.pdf at {pdf_path} ({pdf_path.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PAPER_PDF), help="Output PDF path")
    args = parser.parse_args()

    out_pdf = Path(args.out)
    print(f"Reading {PAPER_MD} ...")
    if not PAPER_MD.exists():
        raise FileNotFoundError(f"paper.md not found: {PAPER_MD}")
    md_text = PAPER_MD.read_text(encoding="utf-8")
    title, abstract = extract_title_abstract(md_text)
    if not title:
        raise ValueError("Could not extract title from paper.md (no '# ' heading found)")
    if not abstract:
        raise ValueError("Could not extract abstract from paper.md (no '> **Abstract.**' block found)")
    print(f"  Title: {title[:80]}...")
    print(f"  Abstract: {len(abstract)} chars")

    body = parse_md_to_latex(md_text)
    print(f"  Body: {len(body)} chars")

    tex_text = build_latex(title, abstract, body)
    PAPER_TEX.write_text(tex_text, encoding="utf-8")
    print(f"Wrote {PAPER_TEX} ({len(tex_text)} chars)")

    compile_xelatex(PAPER_TEX, out_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
