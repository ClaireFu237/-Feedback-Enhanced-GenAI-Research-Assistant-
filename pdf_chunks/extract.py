"""Extract positioned PDF text and conservatively reconstruct reading order."""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter

import pymupdf

VERSION = "positioned-blocks-v1"
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})
KNOWN_HEADING = re.compile(
    r"^(abstract|introduction|background|related work|method(?:s|ology)?|"
    r"experiments?|results?|discussion|conclusions?|limitations?|"
    r"acknowledg(?:e)?ments?|references|bibliography|supplementary material|"
    r"appendi(?:x|ces)(?:\s.*)?)$", re.I
)


def clean_line(text):
    text = unicodedata.normalize("NFC", text.translate(LIGATURES))
    text = "".join(c for c in text if unicodedata.category(c) not in {"Cc", "Cf"} or c.isspace())
    return " ".join(text.split())


def join_lines(lines):
    """Only repair a lowercase word broken at a physical line boundary."""
    result = ""
    for line in lines:
        text = clean_line(line)
        if not text:
            continue
        if re.search(r"[a-z]{3,}-$", result) and re.match(r"^[a-z]{2,}", text):
            result = result[:-1] + text
        else:
            result += (" " if result else "") + text
    return result


def edge_signature(text):
    return re.sub(r"\d+", "#", clean_line(text).casefold())


def reading_order(blocks, width):
    """Order two-column bands around spanning headings/captions when detectable.

    This is a geometric heuristic, not a general table/formula parser.
    """
    # Arrange title/author rows separately from the two-column article body.
    abstract = [b for b in blocks if any(clean_line(l["text"]).casefold() == "abstract" for l in b["lines"])]
    if abstract:
        body_top = min(b["bbox"][1] for b in abstract)
        front = [b for b in blocks if b["bbox"][3] <= body_top]
        if front and len(front) < len(blocks):
            front_ids = {id(b) for b in front}
            return sorted(front, key=lambda b: (b["bbox"][1], b["bbox"][0])) + reading_order(
                [b for b in blocks if id(b) not in front_ids], width)
    middle = width / 2
    left = [b for b in blocks if b["bbox"][2] <= middle + width * .025]
    right = [b for b in blocks if b["bbox"][0] >= middle - width * .025]
    if len(left) < 2 or len(right) < 2 or len(left) + len(right) < len(blocks) * .55:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    narrow_ids = {id(b) for b in left + right}
    spanning = sorted((b for b in blocks if id(b) not in narrow_ids), key=lambda b: b["bbox"][1])
    remaining = [b for b in blocks if id(b) in narrow_ids]
    ordered = []

    def columns(band):
        return sorted(band, key=lambda b: (b["bbox"][0] >= middle - width * .025,
                                           b["bbox"][1], b["bbox"][0]))

    for anchor in spanning:
        before = [b for b in remaining if (b["bbox"][1] + b["bbox"][3]) / 2 < anchor["bbox"][1]]
        ordered.extend(columns(before))
        consumed = {id(b) for b in before}
        remaining = [b for b in remaining if id(b) not in consumed]
        ordered.append(anchor)
    return ordered + columns(remaining)


def is_heading(line, body_size):
    text = clean_line(line["text"])
    if not text or len(text) > 120 or len(text.split()) > 16 or not any(c.isalpha() for c in text):
        return False
    if KNOWN_HEADING.fullmatch(text):
        return True
    numbered = re.match(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+[A-Z]", text)
    # Numbered prose, reference entries and equations should not become headings.
    if numbered and not re.search(r"[.;,]$", text) and len(text.split()) <= 12:
        return line["size"] >= body_size * 1.03 or line["bold"]
    return (line["bold"] and line["size"] >= body_size * 1.08
            and not re.search(r"[.;,]$", text))


def extract_pdf(path, include_references=False):
    pages = []
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDF requires a password")
        if not pdf.page_count:
            raise ValueError("PDF has no pages")
        for number, page in enumerate(pdf, 1):
            blocks = []
            for index, block in enumerate(page.get_text("dict", sort=False)["blocks"]):
                if block["type"] != 0:
                    continue
                lines = []
                for line in block["lines"]:
                    spans = line["spans"]
                    text = "".join(s["text"] for s in spans)
                    if text.strip():
                        lines.append({"text": text, "bbox": list(line["bbox"]),
                                      "size": max(s["size"] for s in spans),
                                      "bold": all(s["flags"] & 16 for s in spans if s["text"].strip())})
                # Some PDFs put a section number and its title into separate
                # lines even though they share a baseline.
                merged = []
                for line in lines:
                    if (merged and abs(merged[-1]["bbox"][1] - line["bbox"][1]) < line["size"] * .3
                            and 0 <= line["bbox"][0] - merged[-1]["bbox"][2] <= line["size"] * 3):
                        previous = merged[-1]
                        previous["text"] += " " + line["text"]
                        previous["bbox"] = [min(previous["bbox"][0], line["bbox"][0]),
                                            min(previous["bbox"][1], line["bbox"][1]),
                                            max(previous["bbox"][2], line["bbox"][2]),
                                            max(previous["bbox"][3], line["bbox"][3])]
                        previous["size"] = max(previous["size"], line["size"])
                        previous["bold"] = previous["bold"] and line["bold"]
                    else:
                        merged.append(line)
                if lines:
                    blocks.append({"block_id": f"p{number}:b{index}", "bbox": list(block["bbox"]),
                                   "raw_text": "\n".join("".join(s["text"] for s in l["spans"])
                                                          for l in block["lines"]), "lines": merged})
            pages.append({"page_number": number, "width": page.rect.width,
                          "height": page.rect.height, "blocks": blocks})

    edge_pages = Counter()
    for page in pages:
        signatures = set()
        for block in page["blocks"]:
            for line in block["lines"]:
                if line["bbox"][3] < page["height"] * .09 or line["bbox"][1] > page["height"] * .91:
                    signatures.add(edge_signature(line["text"]))
        edge_pages.update(signatures)
    repeated = {s for s, n in edge_pages.items() if n >= max(3, len(pages) * .4)}
    # Estimate from prose, so plot labels, tiny references and equations do not
    # make normal body lines look like large-font section headings.
    sizes = [l["size"] for p in pages for b in p["blocks"] for l in b["lines"]
             if len(l["text"]) >= 45 and len(l["text"].split()) >= 5]
    body_size = statistics.median(sizes) if sizes else 10
    paragraphs, pieces, cursor = [], [], 0
    section, section_id, references = "Unsectioned", 0, False
    removed = 0

    def append_paragraph(lines, block, page, heading=False):
        nonlocal cursor, section, section_id, references
        text = join_lines(l["text"] for l in lines)
        if not text:
            return
        if heading:
            section, section_id = text, section_id + 1
            name = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+", "", text).casefold()
            if name in {"references", "bibliography"}:
                references = True
            elif re.match(r"^(?:appendi[xc]|supplement|[a-z](?:\.|\s))", text, re.I):
                references = False
            # Keep bibliography mode until a genuine appendix/supplement heading.
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        paragraphs.append({"block_id": block["block_id"], "page_number": page["page_number"],
                           "bbox": [min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines),
                                    max(l["bbox"][2] for l in lines), max(l["bbox"][3] for l in lines)],
                           "char_start": start, "char_end": cursor, "section": section,
                           "section_id": section_id, "is_heading": heading,
                           "searchable": include_references or not references})

    for page in pages:
        for block in reading_order(page["blocks"], page["width"]):
            pending = []
            for line in block["lines"]:
                edge = line["bbox"][3] < page["height"] * .09 or line["bbox"][1] > page["height"] * .91
                text = clean_line(line["text"])
                numeric = re.fullmatch(r"(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?", text, re.I)
                vertical_margin = line["bbox"][2] < page["width"] * .07 and "arxiv:" in text.casefold()
                if vertical_margin or (edge and (edge_signature(text) in repeated or numeric)):
                    removed += 1
                    continue
                name = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+", "", text)
                preamble = page["page_number"] == 1 and section_id == 0 and not KNOWN_HEADING.fullmatch(name)
                if not preamble and is_heading(line, body_size):
                    append_paragraph(pending, block, page)
                    pending = []
                    append_paragraph([line], block, page, heading=True)
                else:
                    pending.append(line)
            append_paragraph(pending, block, page)
    text = "".join(pieces)
    if not text.strip():
        raise ValueError("No usable text layer; OCR is not included in this pipeline")
    low_pages = [p["page_number"] for p in pages
                 if sum(len(b["raw_text"].strip()) for b in p["blocks"]) < 80]
    report = {"page_count": len(pages), "low_text_pages": low_pages,
              "removed_edge_lines": removed,
              "excluded_reference_paragraphs": sum(not p["searchable"] for p in paragraphs),
              "warnings": (["Low-text pages need visual review; no OCR was performed"] if low_pages else [])}
    return pages, text, paragraphs, report
