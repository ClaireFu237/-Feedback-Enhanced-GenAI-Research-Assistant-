"""Run local retrieval without web or cloud dependencies."""
import argparse
import json
import sys
from .engine import LocalRetriever, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='Dataset key in the selected config; defaults to its default_dataset or first key')
    parser.add_argument('--method', choices=METHODS, default='bm25')
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--document-id', action='append', dest='document_ids')
    parser.add_argument('--config')
    args = parser.parse_args()
    try:
        retriever = LocalRetriever(args.dataset, args.config)
        output = retriever.search(args.query, args.top_k, args.document_ids, args.method)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    output['setup_timings'] = retriever.timings
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
