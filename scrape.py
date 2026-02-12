"""
Stream sentence-level source text from a MediaWiki XML dump.

Usage:
  python3 scrape.py --xml zhwiki-latest-pages-articles.xml --out texts.txt --limit-pages 0

Notes:
- No zh-cn/zh-tw conversion is performed.
- By default, only namespace 0 (article pages) is processed.
- Writes one sentence per line.
- --limit-pages 0 means "no limit".
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys
import xml.etree.ElementTree as ET

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


XML_NS = "http://www.mediawiki.org/xml/export-0.11/"
PAGE_TAG = f"{{{XML_NS}}}page"
LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
EXT_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")
REF_BLOCK_RE = re.compile(r"<ref[^>]*?>.*?</ref>", flags=re.IGNORECASE | re.DOTALL)
REF_SELF_RE = re.compile(r"<ref[^>]*/\s*>", flags=re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
MULTISPACE_RE = re.compile(r"[ \t]+")
CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")


def strip_templates(text: str) -> str:
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == "{" and text[i + 1] == "{":
            depth += 1
            i += 2
            continue
        if i + 1 < n and text[i] == "}" and text[i + 1] == "}":
            if depth > 0:
                depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def decode_wikilinks(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return ""

        parts = inner.split("|")
        target = parts[0].strip()
        display = parts[-1].strip()

        blocked_prefixes = (
            "file:",
            "image:",
            "category:",
            "template:",
            "special:",
            "wikipedia:",
            "help:",
            "module:",
            "portal:",
            "user:",
            "talk:",
        )
        if target.lower().startswith(blocked_prefixes):
            return ""
        if display:
            return display
        return target.split("#", 1)[0]

    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = LINK_RE.sub(repl, cur)
    return cur


def normalize_wikitext(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = REF_BLOCK_RE.sub(" ", text)
    text = REF_SELF_RE.sub(" ", text)
    text = strip_templates(text)
    text = EXT_LINK_RE.sub(r"\2", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = decode_wikilinks(text)

    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^[*#;:]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\{\|.*?$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\|\}.*?$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\|-.*?$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^[!|].*$", " ", text, flags=re.MULTILINE)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.findall(
        r"[^\u3002\uff01\uff1f!?\uff1b;\n]+[\u3002\uff01\uff1f!?\uff1b;]?",
        text,
    )
    return [c.strip() for c in chunks if c.strip()]


def is_valid_sentence(sentence: str, min_chars: int, min_cjk_ratio: float) -> bool:
    chars = [ch for ch in sentence if not ch.isspace()]
    if len(chars) < min_chars:
        return False
    cjk_count = len(CJK_RE.findall(sentence))
    if cjk_count == 0:
        return False
    return (cjk_count / len(chars)) >= min_cjk_ratio


def iter_pages(path: str, limit_pages: int | None, include_all_namespaces: bool):
    count = 0
    context = ET.iterparse(path, events=("end",))
    try:
        for _, elem in context:
            if elem.tag != PAGE_TAG:
                continue

            ns_node = elem.find(f"{{{XML_NS}}}ns")
            ns_value = (ns_node.text or "").strip() if ns_node is not None else ""
            if not include_all_namespaces and ns_value != "0":
                elem.clear()
                count += 1
                if limit_pages is not None and count >= limit_pages:
                    break
                continue

            title_node = elem.find(f"{{{XML_NS}}}title")
            title = (title_node.text or "").strip() if title_node is not None else ""
            text_node = elem.find(f"{{{XML_NS}}}revision/{{{XML_NS}}}text")
            text = (text_node.text or "") if text_node is not None else ""
            yield title, ns_value, text

            elem.clear()
            count += 1
            if limit_pages is not None and count >= limit_pages:
                break
    except ET.ParseError as exc:
        print(
            f"[warn] XML parse stopped early due to malformed/incomplete XML: {exc}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract source sentences from zhwiki XML dump."
    )
    parser.add_argument(
        "--xml",
        default="data/zhwiki/zhwiki-latest-pages-articles.xml",
        help="Path to MediaWiki XML dump",
    )
    parser.add_argument(
        "--out",
        default="texts.txt",
        help="Output path (one source sentence per line)",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=0,
        help="Stop after N pages; 0 means no limit",
    )
    parser.add_argument(
        "--include-all-namespaces",
        action="store_true",
        help="Include non-article namespaces (default is namespace 0 only)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=8,
        help="Minimum non-space chars per sentence",
    )
    parser.add_argument(
        "--min-cjk-ratio",
        type=float,
        default=0.40,
        help="Minimum Chinese character ratio per sentence",
    )
    args = parser.parse_args()

    limit = None if args.limit_pages <= 0 else args.limit_pages
    pages_seen = 0
    sentences_written = 0

    bar = None
    if _TQDM:
        bar = tqdm(
            total=limit if limit is not None else None,
            unit="page",
            desc="pages",
        )

    with io.open(args.out, "w", encoding="utf-8") as out_f:
        for _, _, raw_text in iter_pages(args.xml, limit, args.include_all_namespaces):
            pages_seen += 1
            if bar is not None:
                bar.update(1)

            cleaned = normalize_wikitext(raw_text)
            for sentence in split_sentences(cleaned):
                if not is_valid_sentence(sentence, args.min_chars, args.min_cjk_ratio):
                    continue
                out_f.write(sentence)
                out_f.write("\n")
                sentences_written += 1

    if bar is not None:
        bar.close()

    print(
        f"Wrote {sentences_written} sentences to {args.out} from {pages_seen} pages."
    )


if __name__ == "__main__":
    main()
