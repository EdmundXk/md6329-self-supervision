r"""Produces an Overleaf-ready copy of the report that compiles against the OFFICIAL
NeurIPS style file (neurips_2024.sty), which cannot be redistributed here.

Differences from report/report.tex:
  * \usepackage{neuripsstyle}  ->  \usepackage[final]{neurips_2024}
  * the custom \nipstitle/\nipsauthor block -> standard \title/\author/\maketitle
  * figure paths ../figs/...  ->  figs/...   (flat project layout)
Everything else, including every number, is byte-identical to report.tex.

Output: overleaf/  (upload this folder's contents, or the zip, to Overleaf)
"""
import os, re, shutil, zipfile

SRC = os.path.join("report", "report.tex")
OUT = "overleaf"

TITLE = "When Does Self-Supervision Beat ImageNet? \\\\ Domain Distance Governs " \
        "Label-Efficient Pretraining in Medical Imaging"
AUTHOR = r"""Yang Xingkai \\
  Student ID: G2610153K \\
  MD6329 --- Deep Learning for Healthcare AI"""


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    tex = open(SRC, encoding="utf-8").read()

    # 1. preamble: swap the local layout package for the official style
    tex = tex.replace(r"\usepackage{neuripsstyle}",
                      "\\usepackage[final]{neurips_2024}\n"
                      "% neurips_2024.sty loads natbib and hyperref itself")
    tex = tex.replace("\\usepackage{natbib}\n", "")
    tex = tex.replace("\\usepackage{hyperref}\n", "")

    # 2. title block: the official template uses \title/\author/\maketitle
    block = re.search(r"\\nipstitle\{.*?\}\s*\n\\nipsauthor\{.*?\}\s*\n", tex, re.S)
    tex = (tex[:block.start()]
           + "\\title{%s}\n\n\\author{%s}\n\n\\maketitle\n\n" % (TITLE, AUTHOR)
           + tex[block.end():])
    # \title must sit in the preamble, before \begin{document}
    tex = tex.replace("\\begin{document}\n\n\\title{", "\\title{")
    tex = tex.replace("\\maketitle\n\n\\begin{abstract}",
                      "\\begin{document}\n\n\\maketitle\n\n\\begin{abstract}")

    # 3. flat figure paths
    tex = tex.replace("{../figs/", "{figs/")

    open(os.path.join(OUT, "report.tex"), "w", encoding="utf-8").write(tex)

    # 4. supporting files
    shutil.copy(os.path.join("report", "refs.bib"), OUT)
    for t in sorted(f for f in os.listdir("report") if f.startswith("tab_")):
        shutil.copy(os.path.join("report", t), OUT)
    for fig in ["fig1_data.png", "fig2_label_efficiency.png", "fig3_probe_ablation.png",
                "fig4_analysis.png", "fig5_gradcam.png"]:
        shutil.copy(os.path.join("figs", fig), os.path.join(OUT, "figs"))

    open(os.path.join(OUT, "HOW_TO_COMPILE.txt"), "w", encoding="utf-8").write(
        "Compiling this report on Overleaf\n"
        "================================\n\n"
        "This folder does NOT contain neurips_2024.sty, which NeurIPS distributes\n"
        "separately. Pick either route.\n\n"
        "ROUTE A - start from Overleaf's NeurIPS template (easiest)\n"
        "  1. Open overleaf.com -> New Project -> Templates -> search 'NeurIPS 2024'.\n"
        "  2. Create the project. It already contains neurips_2024.sty.\n"
        "  3. Delete the template's example .tex and .bib files.\n"
        "  4. Upload everything from this folder (report.tex, refs.bib, tab_*.tex,\n"
        "     and the figs/ folder - keep figs/ as a folder, do not flatten it).\n"
        "  5. Menu -> Main document -> report.tex.\n"
        "  6. Compiler must be pdfLaTeX. Click Recompile.\n\n"
        "ROUTE B - bring your own style file\n"
        "  1. Download neurips_2024.sty from https://neurips.cc/Conferences/2024/\n"
        "     (Call for Papers -> LaTeX style files).\n"
        "  2. New Project -> Upload Project -> this folder's zip.\n"
        "  3. Upload neurips_2024.sty into the project root.\n"
        "  4. Set the main document to report.tex and recompile.\n\n"
        "BEFORE SUBMITTING\n"
        "  - The \\author{} block already carries the author name and student ID; add an\n"
        "    email line there only if your submission requires one.\n"
        "  - [final] is already set, so the paper prints with author names visible\n"
        "    (use [preprint] for an anonymous version).\n"
        "  - Bibliography: pdfLaTeX -> BibTeX -> pdfLaTeX twice, or just press\n"
        "    Recompile a second time on Overleaf.\n")

    with zipfile.ZipFile("overleaf_project.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, OUT).replace("\\", "/"))
    print("[overleaf] overleaf/ and overleaf_project.zip written")


if __name__ == "__main__":
    main()
