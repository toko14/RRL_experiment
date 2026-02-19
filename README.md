# ロバスト強化学習の研究（TCRMDP / M2TD3）

ロバスト強化学習（RRL）の研究として、**Time-Constrained Robust MDP (TCRMDP)** と **M2TD3** をベースに、不確実性下での方策学習・評価を行ったリポジトリです。

## 概要

- **TCRMDP** (NeurIPS 2024): 時間制約付きロバスト MDP の公式実装。RRLS が提供する Gymnasium 互換のロバスト環境上で学習・評価します。
- **M2TD3**: 複数の不確実パラメータに対するロバスト方策を学習する手法。TCRMDP と組み合わせて使用しています。
- 卒業研究として、HalfCheetah / Ant / Hopper / Walker 等の MuJoCo 環境で実験し、可視化・分析を行いました。

## リポジトリ構成

| ディレクトリ | 内容 |
|-------------|------|
| `TCRMDP/` | TCRMDP および M2TD3 のコア実装（学習・評価のエントリポイント含む） |
| `scripts/` | RRLS を用いた評価スクリプト、キャッシュ書き換え等 |
| `visualization/` | ポリシー評価・ハイパラスイープ等の Jupyter ノートブック |

## 環境構築

### 1. RRLS 環境のインストール（必須）

TCRMDP は [RRLS](https://github.com/SuReLI/RRLS) が提供するロバスト環境上で動作します。**pip 版は不完全なため、GitHub から clone して editable インストールを推奨します。**

```bash
git clone https://github.com/SuReLI/RRLS
cd RRLS
pip install -e .
```

### 2. 本リポジトリの依存関係

```bash
pip install -r requirements.txt
pip install packaging  # 必要に応じて
```

### 3. 実行時の注意

- **Gymnasium**: RRLS の `pyproject.toml` は `gymnasium==1.0.0a1` を要求しますが、`gymnasium==1.2.1` で動作確認済みです（警告は出るが問題なし）。
- **MuJoCo 描画**: headless 環境や CI では `MUJOCO_GL=egl` を設定してください。
  ```bash
  MUJOCO_GL=egl python your_script.py
  ```
- **import 時の副作用**: RRLS の `evaluate.py` が import 時に MuJoCo 環境を自動生成する場合、ビルドで失敗することがあります。その場合は RRLS 側で該当処理を遅延させる等の対応が必要です。

## 学習・評価の例

TCRMDP のエントリポイントは `TCRMDP/src/` にあります。例:

```bash
cd TCRMDP
python src/main_m2td3.py --help
python src/main_m2td3_only_soft_omega.py --help  # カスタム版
```

評価は `scripts/rrls_light_eval.py` や `visualization/` 内のノートブックから行えます。

## 参考文献

- **RRLS (Benchmark)**: [Robust Reinforcement Learning in Continuous Control (arxiv 2406.08406)](https://arxiv.org/pdf/2406.08406)
- **TCRMDP**: [Time-Constrained Robust MDPs (NeurIPS 2024)](https://openreview.net/forum?id=NKpPnb3YNg)
- **TCRMDP 公式実装**: [SuReLI/TCRMDP](https://github.com/SuReLI/TCRMDP)
- **RRLS**: [SuReLI/RRLS](https://github.com/SuReLI/RRLS)

## License

- **TCRMDP**（本リポジトリの `TCRMDP/`）: **MIT License**  
  Copyright (c) 2024 Supaero Reinforcement Learning Initiative and Institut de recherche technologique Antoine de Saint-Exupéry.

- **RRLS**（環境構築時に別途 clone する依存関係）: **MIT License**  
  Copyright (c) 2023 Supaero Reinforcement Learning Initiative and Institut de recherche technologique Antoine de Saint-Exupéry.

- その他（`scripts/`, `visualization/`, `graduation_thesis/` 等の追加コード・ノートブック）: 本リポジトリのライセンスに従います。

詳細は `docs/THIRD_PARTY_NOTICES.md` を参照してください。
