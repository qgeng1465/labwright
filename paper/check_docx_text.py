# -*- coding: utf-8 -*-
"""Dump every string in the generated report docx (paragraphs + tables) to text.

This is the "TEXT IDENTICAL" audit tool for the report: after regenerating
``paper/report_to_teacher.py``'s docx, dump its text and diff/eyeball it
against the source strings so no rendered number drifts from what the
generator claims. Every number in the docx is then cross-checked by hand
against the committed ``results/*.json`` before the next commit.

Usage::

    python -m paper.check_docx_text            # print to stdout
    python -m paper.check_docx_text -o /tmp/docx.txt
"""

from __future__ import annotations

import argparse
import sys

from docx import Document

_DIR = __file__.rsplit("/", 1)[0]
ROOT = _DIR.rsplit("/", 1)[0]
DEFAULT_OUT = "paper/report_to_teacher.docx.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docx", default=f"{ROOT}/paper/Labwright_工作报告_给导师.docx")
    ap.add_argument("-o", "--out", default=None, help="text file; default stdout")
    args = ap.parse_args()

    doc = Document(args.docx)
    lines: list[str] = []
    # Body paragraphs in document order; tables are re-walked after so their
    # cells read row-by-row.
    body = doc.element.body
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    def emit_table(tbl: Table) -> None:
        for row in tbl.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells))

    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            if p.text.strip():
                lines.append(p.text)
        elif tag == "tbl":
            emit_table(Table(child, doc))

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"dumped {len(lines)} lines -> {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
