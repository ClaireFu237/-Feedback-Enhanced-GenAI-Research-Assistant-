# Dataset Attribution and License

The data files in this directory are redistributed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/). The [HotpotQA project](https://hotpotqa.github.io/) publishes its dataset and processed Wikipedia corpus under this license. This notice concerns the bundled data, not a license assignment for the application code or pretrained models.

Source: HotpotQA, *A Dataset for Diverse, Explainable Multi-hop Question Answering*, Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning, EMNLP 2018. [Paper](https://arxiv.org/abs/1809.09600).

Changes: 100 records were previously sampled from the validation/distractor split with seed 42; their context paragraphs, including distractors, were pooled and deduplicated into 997 documents. Text was joined from provided sentences. This package preserves those prepared paragraph texts, assigns stable project IDs, records character offsets, and makes local text locators portable. It excludes answer/evidence labels. It is a custom shared-corpus demonstration, not an official fullwiki benchmark.

Wikipedia contributors are the source article authors. attribution.jsonl retains article titles, Wikipedia article links and project IDs; id_map.jsonl retains the prepared source IDs. Exact Wikipedia revision IDs were not available in the prepared subset and are not fabricated. No endorsement by HotpotQA or Wikipedia is implied.
