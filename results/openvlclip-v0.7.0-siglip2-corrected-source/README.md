# OpenVLCLIP SigLIP 2 corrected-source validation

> **重要: これは「v0.7.0を基にした未公開の修正ソース」による測定です。正式PyPI版 `openvlclip==0.7.0` の性能ではありません。**
>
> パッケージのversion表示は `0.7.0` のままですが、実際に使用したコードはOpenVLCLIPのGit commit `8596295c1f519506eda5842e0322b2fbc8ad8daa` です。固定Git SHA、依存lock、タグとの差分は[provenance](validation/source-provenance.json)に記録しています。

## 測定結果

FixResとNaFlexの各6条件、計12条件をフルデータセットで測定しました。全runはCUDA FP32、seed 0、requested batch size 16で、`working_tree_dirty=false`です。

- [結果レポート](report.md) — 精度、検索Recall、推論速度、実測GPUメモリ
- [summary CSV](summary.csv)
- [既存モデルとのmatched comparison CSV](model-comparison.csv) — Qwen3-VL-Embedding-2Bを含む既存9モデルとの55件の比較行
- [修正前後のSigLIP 2直接比較 CSV](validation/siglip2-before-after.csv) — 同一12条件のraw間でプロンプト・split・データ順序・checkpoint/tokenizer revisionを確認した比較
- [raw JSON](raw/)
- [lifecycle](validation/result-lifecycle.json)
- [model comparison audit](validation/model-comparison-audit.json)
- [グラフ](plots/)

| Model | CIFAR-10 P0 Top-1 前→後 | CIFAR-10 P1 Top-1 前→後 | CIFAR-100 P0 Top-1 前→後 | CIFAR-100 P1 Top-1 前→後 |
|---|---:|---:|---:|---:|
| SigLIP 2 FixRes | 57.14% → 93.62% (+36.48 pp) | 55.39% → 93.85% (+38.46 pp) | 24.45% → 73.68% (+49.23 pp) | 13.13% → 73.25% (+60.12 pp) |
| SigLIP 2 NaFlex | 55.30% → 94.05% (+38.75 pp) | 78.28% → 93.69% (+15.41 pp) | 29.78% → 75.43% (+45.65 pp) | 25.50% → 73.23% (+47.73 pp) |

修正前後のTop-5とXM3600の両方向Recall差分は[直接比較CSV](validation/siglip2-before-after.csv)にあります。既存のQwen3-VL-Embedding-2Bは比較に利用しましたが、再測定していません。

## 修正内容と128画像の検証

SigLIP 2 FixRes/NaFlexのテキスト推論で、PADから作った`attention_mask`を`get_text_features()`へ渡さないようにしました。トークン入力の検査は維持し、Hugging Face Transformersの公式既定経路に合わせています。公開済みv0.7.0タグ・PyPI配布物は変更していません。

既存の128画像検証では、FixResは67/128から124/128、NaFlexは62/128から118/128へ改善し、修正後embeddingは公式Transformers経路と一致（最大絶対差0.0、次元768）しました。これは小規模な診断結果であり、上記の10,000画像フル測定とは区別しています。

## 固定した再現情報

- OpenVLCLIP base: tag `v0.7.0`, commit `81c8e1df9cb02d6b9f13c4ebda9ac605a54b62eb`
- 未公開修正ソース: commit `8596295c1f519506eda5842e0322b2fbc8ad8daa`（base tagからSigLIP 2の1ファイルのみ変更）
- Benchmark source: `60bb3c6a10fa827e190954b4f74c6dd82b74e2ae`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER; PyTorch 2.14.1+cu130; CUDA 13.0; Transformers 5.20.1; Python 3.12.10
- CIFAR-10: SHA-256 `744e4b130af776f7a1c7a19427963e1f6d1c63984a1bb2bc2ceb4da2ba54c470`
- CIFAR-100: SHA-256 `e5b40f8042fe89ad338c43b208464e0001bcb4ffba8a6827ed0da83562323a1f`
- XM3600: SHA-256 `ed934a9782286a688ca79d4594aec679ca4accb8d738186f986c831711545fce`; `official_first_caption`, 3,600 images/candidates per language

P0 uses the common CLIP Benchmark-compatible templates; P1 uses the separate FixRes and NaFlex model-card photo prompt policies. XM3600 uses P2 in English and Japanese. The precise checkpoint, tokenizer, processor revisions, dataset checksums, lock hash, and source patch are recorded in [source provenance](validation/source-provenance.json) and each raw run.

The report is generated from the validated raw JSON using the existing `vlbench report` workflow. Its source-identity banner is an editorial provenance note; it does not change any measured values.
