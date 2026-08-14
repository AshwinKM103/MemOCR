# Benchmarks

Performance/profiling and manual API-usage scripts. These are **not** unit tests: they require a
running server, GPU, or manual timing inspection, and are not wired into CI. Run them by hand when
investigating a specific performance question.

- `recurrent/perf_*.py` — tokenizer/encoding throughput probes for `recurrent/`.
- `md2img/test_*.py` — manual exercises of the markdown/HTML rendering API servers in `md2img/`
  (`markdown_api_server.py`, `html_api_server.py`). Start the relevant server first, then run the
  script against it.
