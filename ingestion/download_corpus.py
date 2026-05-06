#!/usr/bin/env python3
"""
Download open-access plant biology papers from NCBI PubMed Central.

Uses NCBI E-utilities (free, no API key required for < 3 req/sec).
Saves a JSONL corpus to ingestion/data/corpus.jsonl.

Usage:
    python -m ingestion.download_corpus
    python -m ingestion.download_corpus --max-papers 30 --query "Arabidopsis drought"

NCBI terms of service: https://www.ncbi.nlm.nih.gov/home/about/policies/
Rate limit respected automatically (0.4 sec between requests = 2.5 req/sec < 3 limit).
"""
import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI requests ~0.33 sec between requests for anonymous access
REQUEST_DELAY = 0.4

DATA_DIR = Path(__file__).parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"

DEFAULT_QUERY = (
    "(arabidopsis[Title/Abstract] OR "
    "plant defense[Title/Abstract] OR "
    "crop protection[Title/Abstract]) AND "
    "(systems biology[Title/Abstract] OR "
    "transcriptome[Title/Abstract] OR "
    "metabolomics[Title/Abstract] OR "
    "plant immunity[Title/Abstract])"
)


def search_pmc(query: str, max_results: int = 50) -> list[str]:
    """
    Search PMC Open Access for paper IDs matching the query.
    Returns a list of PMC IDs (strings without the 'PMC' prefix).
    """
    logger.info("Searching PMC for: %s", query[:80])
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query + " AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]
    logger.info("Found %d paper IDs", len(ids))
    return ids


def fetch_paper_xml(pmcid: str) -> str | None:
    """Fetch full-text JATS XML for a single PMC paper."""
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "rettype": "xml", "retmode": "xml"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Failed to fetch PMC%s: %s", pmcid, exc)
        return None


def parse_jats_xml(xml_text: str, pmcid: str) -> dict | None:
    """
    Parse JATS (Journal Article Tag Suite) XML used by PMC.
    Extracts: title, authors, abstract, and body paragraphs.
    Returns None if the XML is malformed or missing key fields.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("XML parse error for PMC%s: %s", pmcid, exc)
        return None

    ns = {"jats": "https://jats.nlm.nih.gov/ns/archiving/1.0/"}

    # ── Title ────────────────────────────────────────────────────────────────
    title_el = root.find(".//article-title")
    title = ("".join(title_el.itertext()).strip() if title_el is not None else "")

    # ── Authors ──────────────────────────────────────────────────────────────
    authors = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.findtext("name/surname") or ""
        given = contrib.findtext("name/given-names") or ""
        if surname:
            authors.append(f"{surname} {given}".strip())
    authors_str = ", ".join(authors[:5])  # cap at 5 to keep payload small
    if len(authors) > 5:
        authors_str += " et al."

    # ── Abstract ─────────────────────────────────────────────────────────────
    abstract_parts = []
    for el in root.findall(".//abstract//*"):
        text = (el.text or "").strip()
        if text:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # ── Body paragraphs ───────────────────────────────────────────────────────
    body_parts = []
    for p in root.findall(".//body//p"):
        text = "".join(p.itertext()).strip()
        if len(text) > 80:  # skip very short fragments (figure captions etc.)
            body_parts.append(text)

    full_text = abstract + "\n\n" + "\n\n".join(body_parts)

    if len(full_text.strip()) < 200:
        logger.debug("Skipping PMC%s: too little text (%d chars)", pmcid, len(full_text))
        return None

    return {
        "pmcid": pmcid,
        "title": title or f"PMC{pmcid}",
        "authors": authors_str,
        "abstract": abstract,
        "full_text": full_text,
        "char_count": len(full_text),
    }


def download_corpus(query: str = DEFAULT_QUERY, max_papers: int = 50) -> int:
    """
    Main entry point. Downloads papers, parses them, and writes JSONL.
    Returns the number of papers successfully saved.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If corpus already exists, skip (re-run with --force to overwrite)
    if CORPUS_PATH.exists():
        existing = sum(1 for _ in CORPUS_PATH.open())
        if existing > 0:
            logger.info(
                "Corpus already exists with %d papers at %s. "
                "Delete the file or run with --force to re-download.",
                existing,
                CORPUS_PATH,
            )
            return existing

    pmc_ids = search_pmc(query, max_results=max_papers)
    time.sleep(REQUEST_DELAY)

    saved = 0
    with CORPUS_PATH.open("w", encoding="utf-8") as fout:
        for pmcid in tqdm(pmc_ids, desc="Downloading papers"):
            xml_text = fetch_paper_xml(pmcid)
            time.sleep(REQUEST_DELAY)  # respect NCBI rate limit

            if xml_text is None:
                continue

            paper = parse_jats_xml(xml_text, pmcid)
            if paper is None:
                continue

            fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
            saved += 1
            logger.debug("Saved PMC%s: '%s' (%d chars)", pmcid, paper["title"][:60], paper["char_count"])

    logger.info("Corpus complete: %d/%d papers saved to %s", saved, len(pmc_ids), CORPUS_PATH)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download plant biology corpus from PMC OA")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--force", action="store_true", help="Re-download even if corpus exists")
    args = parser.parse_args()

    if args.force and CORPUS_PATH.exists():
        CORPUS_PATH.unlink()
        logger.info("Removed existing corpus (--force)")

    n = download_corpus(query=args.query, max_papers=args.max_papers)
    print(f"\nDone. {n} papers ready for ingestion.")
    print(f"Next step: python -m ingestion.chunk_and_embed")
