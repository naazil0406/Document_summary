"""
Quick CLI to view how a .json file gets chunked, without touching
embeddings/Qdrant. Usage:

    python3 show_chunks.py path/to/your.json
    python3 show_chunks.py path/to/your.json --full     # print full chunk text
    python3 show_chunks.py path/to/your.json --max-chars 2000
"""
import sys
import argparse

from services.json_parser import JSONParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to the .json file")
    ap.add_argument("--max-chars", type=int, default=None, help="Override JSON_CHUNK_MAX_CHARS")
    ap.add_argument("--full", action="store_true", help="Print full chunk text (default: first 300 chars)")
    args = ap.parse_args()

    kwargs = {}
    if args.max_chars:
        kwargs["max_chunk_chars"] = args.max_chars

    parser = JSONParser(**kwargs)
    pages = parser.extract_pages(args.json_path)

    if not pages:
        print("No chunks produced (file unreadable or empty).")
        return

    sizes = [len(p.text) for p in pages]
    print("=" * 70)
    print(f"File: {args.json_path}")
    print(f"Total chunks: {len(pages)}")
    print(f"Chunk sizes -> max: {max(sizes)}  min: {min(sizes)}  avg: {sum(sizes)//len(sizes)}")
    print("=" * 70)

    for i, page in enumerate(pages, start=1):
        print(f"\n--- Chunk {i}/{len(pages)}  ({len(page.text)} chars)  metadata={page.metadata} ---")
        if args.full:
            print(page.text)
        else:
            print(page.text[:300] + ("..." if len(page.text) > 300 else ""))


if __name__ == "__main__":
    main()