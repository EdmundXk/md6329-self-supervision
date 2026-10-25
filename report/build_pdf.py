"""Renders report/report.tex to PDF with ReportLab.

No TeX engine can run in this sandbox (tectonic cannot resolve Windows known-folder
paths from a zero-capability container), so this renders the LaTeX subset actually
used by report.tex: sections, paragraphs, inline markup, figures, \input-ed tabulars,
one display equation, citations and a bibliography. report.tex remains the canonical
source and compiles unchanged on Overleaf with the official NeurIPS style.
"""
import re, os, sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, KeepTogether)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- bibliography
def load_bib(path):
    """Brace-counting parser: entries may end with '}}' on the same line as the
    last field, which a newline-anchored regex misses."""
    txt = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@\w+\s*\{", txt):
        i = m.end()
        depth = 1
        while i < len(txt) and depth:
            depth += (txt[i] == "{") - (txt[i] == "}")
            i += 1
        inner = txt[m.end():i - 1]
        key = inner.split(",", 1)[0].strip()
        body = inner.split(",", 1)[1] if "," in inner else ""
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            j = fm.end(); d = 1
            while j < len(body) and d:
                d += (body[j] == "{") - (body[j] == "}")
                j += 1
            f[fm.group(1).lower()] = " ".join(body[fm.end():j - 1].split())
        entries[key] = f
    return entries


def format_entry(f):
    a = f.get("author", "")
    a = a.replace(" and others", " et al.").replace(" and ", ", ")
    parts = [a, f.get("title", "")]
    venue = f.get("journal") or f.get("booktitle") or ""
    if venue:
        parts.append("<i>%s</i>" % venue)
    if f.get("volume"):
        parts.append("vol. " + f["volume"])
    if f.get("pages"):
        parts.append("pp. " + f["pages"].replace("--", "-"))
    if f.get("year"):
        parts.append(f["year"])
    return ". ".join(p for p in parts if p) + "."


# ---------------------------------------------------------------- inline markup
GREEK = {r"\tau": "\u03c4", r"\sigma": "\u03c3", r"\pm": "\u00b1", r"\times": "\u00d7",
         r"\rightarrow": "\u2192", r"\emph": "", r"\approx": "\u2248", r"\leq": "\u2264",
         r"\geq": "\u2265", r"\in": "\u2208", r"\neq": "\u2260", r"\lVert": "||",
         r"\rVert": "||", r"\top": "T", r"\cdot": "\u00b7", r"\mathcal{L}": "L",
         r"\mathrm{sim}": "sim", r"\exp": "exp", r"\log": "log", r"\sum": "\u03a3",
         r"\frac": "", r"\quad": "  ", r"\qquad": "    "}


def inline(s, cites, bib):
    s = s.replace("\n", " ")
    s = re.sub(r"\\citet\{([^}]*)\}",
               lambda m: _citet(m.group(1), cites, bib), s)
    s = re.sub(r"\\(?:citep|cite)\{([^}]*)\}", lambda m: _cite(m.group(1), cites, bib), s)
    s = re.sub(r"\\(?:textbf|textit)\{(.*?)\}", lambda m: "<b>%s</b>" % m.group(1), s)
    s = re.sub(r"\\emph\{(.*?)\}", lambda m: "<i>%s</i>" % m.group(1), s)
    s = re.sub(r"\\texttt\{(.*?)\}", lambda m: "<font face='Courier'>%s</font>" % m.group(1), s)
    s = re.sub(r"\$([^$]*)\$", lambda m: _math(m.group(1)), s)
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("---", "\u2014").replace("--", "\u2013").replace("~", " ")
    s = s.replace(r"\%", "%").replace(r"\&", "&amp;").replace(r"\_", "_")
    s = s.replace(r"\,", "\u2009").replace(r"\;", " ").replace(r"\ ", " ")
    s = re.sub(r"\\(?:paragraph|section|subsection)\*?\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s?", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _math(m):
    for k, v in GREEK.items():
        m = m.replace(k, v)
    m = re.sub(r"\^\{?\\?\*?\}?", "*", m) if m.strip() in ("^{*}", "^*") else m
    m = re.sub(r"_\{?(\w+)\}?", lambda x: "<sub>%s</sub>" % x.group(1), m)
    m = re.sub(r"\^\{?([\w\-+.]+)\}?", lambda x: "<super>%s</super>" % x.group(1), m)
    m = m.replace("\\", "").replace("{", "").replace("}", "")
    return "<i>%s</i>" % m


def _citet(keys, cites, bib):
    """Textual citation: surname of the first author plus the bracketed number."""
    k = keys.split(",")[0].strip()
    first = bib.get(k, {}).get("author", "").split(" and ")[0]
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1] if first else "?"
    multi = len(bib.get(k, {}).get("author", "").split(" and ")) > 1
    return "%s%s %s" % (surname, " et al." if multi else "", _cite(k, cites, bib))


def _cite(keys, cites, bib):
    out = []
    for k in [k.strip() for k in keys.split(",")]:
        if k not in cites:
            cites.append(k)
        out.append(str(cites.index(k) + 1))
    return "[%s]" % ",".join(out)


# ---------------------------------------------------------------- tabular parsing
def parse_tabular(path, cites, bib):
    txt = open(path, encoding="utf-8").read()
    rows, spans, styles = [], [], []
    body = txt.split(r"\begin{tabular}", 1)[1]
    body = body.split("}", 1)[1].rsplit(r"\end{tabular}", 1)[0]
    r = 0
    for raw in body.split(r"\\"):
        line = raw.strip()
        if not line:
            continue
        rules = re.findall(r"\\(toprule|midrule|bottomrule|cmidrule)", line)
        line = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", line)
        line = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", line).strip()
        for rule in rules:
            if rule == "toprule":
                styles.append(("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black))
            elif rule == "midrule" and r > 0:
                styles.append(("LINEABOVE", (0, r), (-1, r), 0.4, colors.black))
            elif rule == "bottomrule" and r > 0:
                styles.append(("LINEBELOW", (0, r - 1), (-1, r - 1), 0.8, colors.black))
        if not line:
            continue
        cells = []
        for c in line.split("&"):
            c = c.strip()
            mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c)
            if mc:
                n = int(mc.group(1))
                cells.append(inline(mc.group(2), cites, bib))
                spans.append(("SPAN", (len(cells) - 1, r), (len(cells) - 1 + n - 1, r)))
                cells += [""] * (n - 1)
            else:
                cells.append(inline(c, cites, bib))
        rows.append(cells)
        r += 1
    w = max(len(x) for x in rows)
    rows = [x + [""] * (w - len(x)) for x in rows]
    return rows, styles + spans


# ---------------------------------------------------------------- main build
def build():
    src = open(os.path.join(ROOT, "report.tex"), encoding="utf-8").read()
    bib = load_bib(os.path.join(ROOT, "refs.bib"))
    cites = []
    body = src.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]

    base = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.4, leading=12.4,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=base, fontName="Times-Bold", fontSize=12.5,
                        leading=15, spaceBefore=11, spaceAfter=4, alignment=0)
    h2 = ParagraphStyle("h2", parent=base, fontName="Times-Bold", fontSize=10.4,
                        leading=13, spaceBefore=8, spaceAfter=3, alignment=0)
    cap = ParagraphStyle("cap", parent=base, fontSize=8.3, leading=10.4, spaceBefore=4)
    title = ParagraphStyle("t", parent=base, fontName="Times-Bold", fontSize=16.5,
                           leading=20, alignment=TA_CENTER, spaceAfter=3)
    sub = ParagraphStyle("s", parent=base, fontName="Times-Bold", fontSize=11.5,
                         leading=14, alignment=TA_CENTER, spaceAfter=9)
    auth = ParagraphStyle("a", parent=base, fontSize=10, leading=13.5,
                          alignment=TA_CENTER, spaceAfter=11)
    absh = ParagraphStyle("ah", parent=base, fontName="Times-Bold", fontSize=10.4,
                          alignment=TA_CENTER, spaceAfter=4)
    absb = ParagraphStyle("ab", parent=base, fontSize=9.0, leading=11.6,
                          leftIndent=24, rightIndent=24)
    eqs = ParagraphStyle("eq", parent=base, alignment=TA_CENTER, spaceBefore=5,
                         spaceAfter=6, fontSize=9.4)

    story = []
    secno = [0, 0]
    fignum, tabnum, labels = 0, 0, {}
    # pre-scan labels so \ref resolves
    for m in re.finditer(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", body, re.S):
        lm = re.search(r"\\label\{([^}]*)\}", m.group(2))
        if lm:
            if m.group(1) == "figure":
                fignum += 1; labels[lm.group(1)] = str(fignum)
            else:
                tabnum += 1; labels[lm.group(1)] = str(tabnum)
    # section/subsection labels resolve to their numbers
    sa = sb = 0
    for m in re.finditer(r"\\(section\*?|subsection)\{[^}]*\}|\\label\{(sec:[^}]*)\}", body):
        if m.group(2):
            labels[m.group(2)] = "%d.%d" % (sa, sb) if sb else str(sa)
        elif m.group(1) == "section":
            sa += 1; sb = 0
        elif m.group(1) == "subsection":
            sb += 1
    fignum = tabnum = 0

    def resolve(s):
        return re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), s)

    # title block
    t = re.search(r"\\nipstitle\{(.*?)\}\s*\n\\nipsauthor", body, re.S).group(1)
    tparts = t.split(r"\\[0.2em]")
    story.append(Paragraph(inline(tparts[0], cites, bib), title))
    if len(tparts) > 1:
        story.append(Paragraph(inline(tparts[1].replace(r"\large", ""), cites, bib), sub))
    a = re.search(r"\\nipsauthor\{(.*?)\}\s*\n", body, re.S).group(1)
    story.append(Paragraph(inline(a.replace(r"\\", "<br/>"), cites, bib), auth))
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
    story.append(Paragraph("Abstract", absh))
    story.append(Paragraph(inline(ab, cites, bib), absb))
    story.append(Spacer(1, 8))

    rest = body.split(r"\end{abstract}", 1)[1]
    # tokenise into blocks
    pattern = re.compile(
        r"(\\section\*?\{.*?\}|\\subsection\{.*?\}|"
        r"\\begin\{figure\}.*?\\end\{figure\}|"
        r"\\begin\{table\}.*?\\end\{table\}|"
        r"\\begin\{equation\}.*?\\end\{equation\}|"
        r"\\bibliography\{.*?\})", re.S)
    for chunk in pattern.split(rest):
        if not chunk or not chunk.strip():
            continue
        c = chunk.strip()
        if c.startswith(r"\section"):
            name = inline(re.search(r"\{(.*)\}", c, re.S).group(1), cites, bib)
            if not c.startswith(r"\section*"):
                secno[0] += 1; secno[1] = 0
                name = "%d  %s" % (secno[0], name)
            story.append(Paragraph(name, h1))
        elif c.startswith(r"\subsection"):
            secno[1] += 1
            name = inline(re.search(r"\{(.*)\}", c, re.S).group(1), cites, bib)
            story.append(Paragraph("%d.%d  %s" % (secno[0], secno[1], name), h2))
        elif c.startswith(r"\begin{equation}"):
            eq = c.replace(r"\begin{equation}", "").replace(r"\end{equation}", "")
            story.append(Paragraph(_math(eq.replace("\n", " ")), eqs))
        elif c.startswith(r"\begin{figure}"):
            fignum += 1
            img = re.search(r"\\includegraphics\[width=([\d.]*)\\textwidth\]\{(.*?)\}", c)
            capm = re.search(r"\\caption\{(.*?)\}\s*\n?\s*\\label", c, re.S)
            flow = []
            if img:
                p = os.path.normpath(os.path.join(ROOT, img.group(2)))
                from PIL import Image as PILImage
                iw, ih = PILImage.open(p).size
                frac = float(img.group(1)) if img.group(1) else 1.0   # bare \textwidth
                w = frac * 5.5 * inch
                flow.append(Image(p, width=w, height=w * ih / iw))
            if capm:
                flow.append(Paragraph("<b>Figure %d:</b> %s" % (
                    fignum, inline(resolve(capm.group(1)), cites, bib)), cap))
            story.append(Spacer(1, 5)); story.extend(flow); story.append(Spacer(1, 7))
        elif c.startswith(r"\begin{table}"):
            tabnum += 1
            capm = re.search(r"\\caption\{(.*?)\}\s*\n?\s*\\label", c, re.S)
            inp = re.search(r"\\input\{(.*?)\}", c)
            flow = [Paragraph("<b>Table %d:</b> %s" % (
                tabnum, inline(resolve(capm.group(1)), cites, bib)), cap)] if capm else []
            if inp:
                rows, sty = parse_tabular(os.path.join(ROOT, inp.group(1)), cites, bib)
                fs = 6.6 if len(rows[0]) > 5 else 7.4
                data = [[Paragraph(x, ParagraphStyle("c", fontName="Times-Roman",
                                                     fontSize=fs, leading=fs + 1.8,
                                                     alignment=1 if j else 0)) for j, x in enumerate(r)]
                        for r in rows]
                t = Table(data, hAlign="CENTER")
                t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 3),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 3)] + sty))
                flow.append(Spacer(1, 3)); flow.append(t)
            story.append(Spacer(1, 5)); story.extend(flow); story.append(Spacer(1, 7))
        elif c.startswith(r"\bibliography"):
            story.append(Paragraph("References", h1))
            for i, k in enumerate(cites, 1):
                story.append(Paragraph("[%d] %s" % (i, format_entry(bib[k])),
                                       ParagraphStyle("r", parent=base, fontSize=8.3,
                                                      leading=10.3, leftIndent=16,
                                                      firstLineIndent=-16, spaceAfter=2.4)))
        else:
            for para in re.split(r"\n\s*\n", c):
                para = para.strip()
                if not para:
                    continue
                pm = re.match(r"\\paragraph\{(.*?)\}(.*)", para, re.S)
                if pm:
                    txt = "<b>%s</b> %s" % (inline(pm.group(1), cites, bib),
                                            inline(resolve(pm.group(2)), cites, bib))
                else:
                    txt = inline(resolve(para), cites, bib)
                if txt:
                    story.append(Paragraph(txt, base))

    doc = SimpleDocTemplate(os.path.join(ROOT, "report.pdf"), pagesize=letter,
                            leftMargin=1.45 * inch, rightMargin=1.45 * inch,
                            topMargin=1.0 * inch, bottomMargin=1.0 * inch,
                            title="When Does Self-Supervision Beat ImageNet?")
    doc.build(story)
    print("[pdf] report.pdf written with %d references" % len(cites))


if __name__ == "__main__":
    build()
