#!/usr/bin/env python3
"""Explicitly prepare pinned pretrained models; never download datasets."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'config/demo_retrieval.json')
    parser.add_argument('--method', choices=['dense','hybrid','hybrid_reranker'], default='hybrid_reranker')
    parser.add_argument('--local-only', action='store_true', help='Verify existing model files without network access')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from services.retrieval.engine import model_cache_root, resolve_model
    config=json.loads(args.config.read_text())
    cache=model_cache_root(config.get('model_cache_root'))
    names=['dense','reranker'] if args.method=='hybrid_reranker' else ['dense']
    for name in names:
        spec=config[name]
        print(f'Preparing {name}: {spec["model_id"]} @ {spec["revision"]}',file=sys.stderr)
        snapshot_download(repo_id=spec['model_id'],revision=spec['revision'],cache_dir=str(cache),
            local_files_only=args.local_only,
            allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json',
                            'special_tokens_map.json','vocab.txt','sentencepiece.bpe.model'])
        path=resolve_model(spec,config.get('model_cache_root'))
        if not (path/'tokenizer_config.json').exists():
            raise FileNotFoundError('Incomplete tokenizer snapshot: '+str(path))
        print(str(path))


if __name__=='__main__':
    main()
