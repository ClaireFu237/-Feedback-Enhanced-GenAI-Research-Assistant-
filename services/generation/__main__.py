"""Run one grounded answer with a required, versioned retrieval profile."""
import argparse
import json
import sys
from pathlib import Path

from .service import create_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--query', required=True)
    parser.add_argument('--document-id', action='append', dest='document_ids')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        service = create_service(profile_path=args.profile, config_path=args.config, output_root=args.output)
        result = service.answer(args.query, args.document_ids)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ('answer_id','retrieval_id','retrieval_profile','model_id',
        'answer','usage','timings','context_source_count')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
