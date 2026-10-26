"""Run local retrieval without web or cloud dependencies."""
import argparse
import json
from .engine import LocalRetriever, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['scifact', 'hotpotqa'], default='scifact')
    parser.add_argument('--method', choices=METHODS, default='bm25')
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--document-id', action='append', dest='document_ids')
    parser.add_argument('--config')
    args = parser.parse_args()
    retriever = LocalRetriever(args.dataset, args.config)
    output = retriever.search(args.query, args.top_k, args.document_ids, args.method)
    output['setup_timings'] = retriever.timings
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
