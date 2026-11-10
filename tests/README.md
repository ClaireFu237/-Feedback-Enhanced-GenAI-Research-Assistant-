# Functional tests

Install `requirements.txt`, then run:

```bash
python -m unittest discover -s tests -p 'test_web_api.py' -v
```

Tests use temporary directories and small generated PDFs. They check stage
boundaries, full pipeline/reuse, source positions, invalidation, text import,
isolation, errors, concurrent requests and restart recovery. No model download is needed. See ../WEB_GUIDE.md for Swagger UI instructions.
