# final_assignment_advanced1
Physical AI応用講座１の最終課題用レポジトリ

Gazebo 上のロボットアーム **CRANE+ V2** に、自作の **SmolVLA（Vision-Language-Action, Flow Matching）** ポリシーで
「赤いブロックを掴んで、緑の置き場に置く」タスクを実行させる。


https://github.com/user-attachments/assets/2d236064-803f-456b-887b-dfba2da7039a


- 動作の様子: `video/pick_place_2x.mp4`（2 倍速・1 分 40 秒。ポリシーの推論だけで動かしている）
- レポート: `report/report.pdf`（原稿は `report/report.html`）

## 目次
- [タスク](#タスク)
- [使用した技術](#使用した技術)
- [システム構成](#システム構成)
- [ディレクトリ構成](#ディレクトリ構成)
- [データの流れ（どのファイルが何を読み、何を作るか）](#データの流れどのファイルが何を読み何を作るか)
- [各ファイルの説明](#各ファイルの説明)
- [実行方法](#実行方法)
- [pixi のタスク一覧](#pixi-のタスク一覧)
- [主な工夫](#主な工夫)

## タスク
| 項目 | 内容 |
|---|---|
| ロボット | CRANE+ V2（RT Corporation 公式 ROS 2 パッケージ `crane_plus`）、Gazebo（Ignition Fortress）上 |
| タスク | テーブル上の固定位置 (0, −0.1) の赤ブロック（5cm 角）を掴み、固定位置 (0, +0.1) の緑の置き場に置く |
| タスク文 | `pick up the red block and place it on the green target` |
| 観測 | 俯瞰カメラ画像（256×256）1 枚 + 関節角 5 次元 `[joint1, joint2, joint3, joint4, gripper]` |
| 行動 | 関節角の目標値 5 次元 × 10 ステップ（action chunk。10fps で 1 秒分） |
| 制約 | 学習は Google Colab 無料枠の T4 で 30 分以内、推論は Mac の CPU のみ |

## 使用した技術
| 層 | 技術 |
|---|---|
| ロボット・環境 | ROS 2 Humble、Gazebo（Ignition Fortress）、RViz / MoveIt 2、ros2_control、rosbag2 |
| データ | rosbags（ROS なしで rosbag を読む）、LeRobot v0.6.0 の LeRobotDataset |
| ポリシー | SmolVLA（講義ノートブックの自作実装）: SmolVLM2-500M（凍結）+ Flow Matching の action head（Transformer 4 層・hidden 256） |
| 学習 | PyTorch、Google Colab（T4 GPU） |
| 実行環境 | Docker（`airobotbook/ros2-desktop-ai-robot-book-humble`）、pixi（Python 環境とタスク）、noVNC |

## システム構成
```
 ① シミュレーション           ② デモ収集                  ③ データ変換
 Gazebo + CRANE+ V2   ──▶   demo.py（固定の動き）  ──▶   rosbag2lerobot.py
 赤ブロック・置き場・           を 25 回実行し              直前値保持で 10fps に同期
 俯瞰カメラ                    ros2 bag record で記録       → LeRobotDataset（7150 フレーム）
        ▲                                                         │ zip でアップロード
        │ 関節指令（/crane_plus_*_controller/joint_trajectory）     ▼
 ⑤ 推論（コンテナの CPU）                              ④ 学習（Colab T4, 25 分）
 infer.py: 画像と /joint_states から  ◀── checkpoint ──  train.ipynb: SmolVLA の action head を
 1 秒分の行動を推論して publish                          Flow Matching で学習
```

推論の流れ（`infer.py`）:
1. 起動時にブロックを初期位置に戻し、アームを初期姿勢にする
2. 最新の俯瞰画像・関節角・タスク文から、SmolVLA で 10 ステップ分の関節角を推論する（Euler 法 10 回）
3. 10 ステップを 1 本の軌道としてアームとグリッパのコントローラに送り、実行し終わったら 2 に戻る

デモ収集と推論で、データがどのトピックを通って流れるかは、末尾の「備考」にある「ROS 2 の仕組み（Gazebo・RViz・トピック）」を参照。

## ディレクトリ構成
`*` は git 管理外（実行中に作られるもの・大きいもの）。

```
.
├── README.md                       # このファイル（第三者向けの説明）
├── pixi.toml                       # Python 環境（convert / infer）とタスク定義
├── pixi.lock                       # pixi が作る、パッケージの版の記録
├── .gitignore
├── config/
│   └── sim.yaml                    # 関節名・トピック・タスク文・fps・デモの動き（共通の設定）
├── model.py                        # SmolVLA + Flow Matching のモデル定義（学習と推論で共通）
├── scripts/
│   └── rosbag2lerobot.py           # rosbag → LeRobotDataset の変換
├── notebooks/
│   └── train.ipynb                 # Colab T4 での学習ノートブック
├── ros2_ws/                        # ROS 2 のワークスペース
│   ├── src/crane_pp/               # 自作 ROS 2 パッケージ（ament_python）
│   │   ├── launch/
│   │   │   └── sim.launch.py       # Gazebo + RViz の起動（pixi run sim）
│   │   │                           #   Gazebo: ExecuteProcess で `ign gazebo -r pick_place.sdf` を実行
│   │   │                           #   RViz: include した公式の run_move_group.launch.py の中で MoveIt と一緒に起動
│   │   │                           #   最後の LaunchDescription([...]) に並べたものが同時に起動する
│   │   ├── worlds/
│   │   │   └── pick_place.sdf      # テーブル・赤ブロック・緑の置き場・俯瞰カメラ
│   │   ├── crane_pp/
│   │   │   ├── __init__.py
│   │   │   ├── demo.py             # デモの動きを流すノード（学習データ集め）
│   │   │   └── infer.py            # 学習済みポリシーで動かすノード
│   │   ├── resource/crane_pp       # ament のパッケージ登録用（空ファイル）
│   │   ├── package.xml             # パッケージ情報・依存
│   │   ├── setup.py                # インストール内容と ros2 run のコマンド登録
│   │   └── setup.cfg               # スクリプトのインストール先
│   └── build/ install/ log/  *     # pixi run build（colcon build）の出力
├── data/  *
│   ├── raw/ep0〜ep24/              # pixi run record が記録した rosbag（1 エピソード = 1 フォルダ）
│   └── interim/crane_pp/           # pixi run convert が作った LeRobotDataset（data / images / meta / videos）
├── crane_pp_dataset.zip  *         # pixi run zip-dataset が作る、Colab にアップロードする zip
├── checkpoints/  *
│   └── smolvla_ckpt.pt             # Colab で学習した重み（train.ipynb がダウンロードさせる）
├── video/  *
│   ├── raw.mov                     # Mac の画面収録（元動画）
│   └── pick_place_2x.mp4           # 提出動画（pixi run video で作る 2 倍速版）
├── report/  *
│   ├── report.html                 # レポートの原稿
│   └── report.pdf                  # 提出用レポート（report.html を Chrome で PDF にしたもの）
├── __pycache__/  *                 # model.py を import したときに Python が作るキャッシュ
└── .pixi/  *                       # pixi が作る Python 環境の実体
```

## データの流れ（どのファイルが何を読み、何を作るか）
| 段階 | 実行するもの | 読むもの | 作るもの |
|---|---|---|---|
| ① シミュレーション | `sim.launch.py`（`pixi run sim`） | `pick_place.sdf`、イメージ内の crane_plus | Gazebo・RViz の画面、トピック `/joint_states`・`/image_raw`・`/clock` |
| ② デモ収集 | `demo.py` + `ros2 bag record`（`pixi run record`） | `config/sim.yaml` のウェイポイント | `data/raw/ep0〜ep24`（rosbag。1 回のデモ = 1 フォルダ） |
| ③ データ変換 | `scripts/rosbag2lerobot.py`（`pixi run convert`） | `data/raw/`、`config/sim.yaml` | `data/interim/crane_pp`（LeRobotDataset）→ zip にして `crane_pp_dataset.zip` |
| ④ 学習 | `notebooks/train.ipynb`（Colab） | `crane_pp_dataset.zip`、`model.py` | `checkpoints/smolvla_ckpt.pt`（学習した重みと正規化の統計量） |
| ⑤ 推論 | `infer.py`（`pixi run infer`） | `smolvla_ckpt.pt`、`model.py`、`config/sim.yaml`、トピック | アーム・グリッパへの関節指令 |
| ⑥ 動画 | ffmpeg（`pixi run video`） | `video/raw.mov`（Mac の画面収録） | `video/pick_place_2x.mp4` |

## 各ファイルの説明

### 設定・環境
#### `config/sim.yaml` — 共通の設定
デモ収集（`demo.py`）・変換（`rosbag2lerobot.py`）・推論（`infer.py`）の 3 つが読む設定ファイル。同じ値を何か所にも書かないよう、ここに集めている。
- `block_pose`: 赤ブロックの初期位置。デモと推論の開始時に、ブロックをこの位置へ戻す
- `arm_joints` / `gripper_joint`: 関節名。state・action は必ずこの順 `[joint1, joint2, joint3, joint4, gripper]` に並べる（`/joint_states` の並び順は当てにならないため）
- `arm_topic` / `gripper_topic` / `joint_states_topic` / `image_topic`: 指令・関節状態・カメラ画像のトピック名
- `task` / `fps` / `chunk_size`: VLM に渡すタスク文、データセットの周波数（10Hz）、1 回の推論で出す行動の数（10 = 1 秒分）
- `gripper_open` / `gripper_close`: グリッパの開閉角
- `waypoints`: デモの動き。1 行が `[joint1〜4, gripper, 到達までの秒数]` で、「初期姿勢 → ブロックの真上 → 下降 → 閉じる → 持ち上げ → 置き場の真上 → 下降 → 開く → 退避 → 初期姿勢」の 10 個。関節角は URDF の寸法から逆運動学で計算した

#### `pixi.toml` / `pixi.lock` — Python 環境とコマンド
- コマンド（task）をまとめている。`pixi run build` などで呼ぶ（一覧は下の「pixi のタスク一覧」）
- Python 環境を 2 つ定義している。LeRobot v0.6.0 は Python 3.12 以上、コンテナの ROS 2 Humble（rclpy）は Python 3.10 用で、1 つの環境に同居できないため
  - `convert`: Python 3.12 + LeRobot v0.6.0 + rosbags（rosbag を ROS なしで読む）。データ変換と zip 作成に使う
  - `infer`: Python 3.10 + PyTorch（CPU 版）+ transformers。推論ノードに使う
- ROS 2 本体（`ros2 launch`・`colcon`・`ros2 bag`）は pixi ではなく、コンテナに apt で入っているものを使う
- `pixi.lock` は pixi が自動で作る、入ったパッケージの版の記録（手で編集しない）

### ROS 2 パッケージ `ros2_ws/src/crane_pp/`
`pixi run build`（colcon build）でビルドすると、`ros2 launch crane_pp ...` や `ros2 run crane_pp demo` で呼べるようになる。

#### `launch/sim.launch.py` — シミュレーションの起動
Gazebo・ロボット・コントローラ・RViz を一度に起動する。コンテナ内の公式ファイル `crane_plus_gazebo/launch/crane_plus_with_table.launch.py` を複製し、次の 2 点だけ変えている（イメージ内の版は world が固定で、カメラ画像の中継もないため include では足りなかった）。
1. world を自作の `pick_place.sdf` にする
2. `ros_gz_bridge` で、Gazebo のカメラ画像を ROS のトピック `/image_raw` に中継する

起動するものとその役割は、末尾の「備考」の「pixi run sim で起動するもの」を参照。

#### `worlds/pick_place.sdf` — シミュレーションの世界
Gazebo に置く物を定義する。公式の `table.sdf` をもとに次を追加した。
- `red_block`: 掴む対象の赤い立方体（5cm 角）。位置 (0, −0.1)。大きさ・質量・摩擦は公式サンプルで掴めることが確認されている箱と同じ
- `green_target`: 置き場の目印の緑の板（8cm 角・厚さ 2mm）。位置 (0, +0.1)。見た目だけで当たり判定はない
- `top_camera`: テーブルの 0.6m 上から真下を向く俯瞰カメラ（256×256、10Hz）。この画像が学習・推論の入力になる。見た目のないモデルなので Gazebo の画面には映らない
- 物理・センサ用のプラグイン、太陽・地面・テーブル（公式と同じ）

#### `crane_pp/demo.py` — デモの動きを流すノード（学習データ集め用）
人がテレオペする代わりに、決まった動きでピック&プレースをする。
1. 起動時に Gazebo のサービス `set_pose` で赤ブロックを初期位置に戻す（毎回同じ条件から始めるため）
2. `config/sim.yaml` のウェイポイントの間を線形補間し、20Hz の関節指令の列（410 個 = 20.5 秒）を作る
3. 指令を 1 個ずつアームとグリッパのトピックに送る。全部送ったら終了する
- 時刻はシム時間（`use_sim_time`）で数える。シムが実時間より遅くても、動きの速さがデモごとに変わらないようにするため
- `pixi run record` は、`ros2 bag record` で 4 トピックを記録しながらこれを 25 回流す
- `reset_block()` と `CONFIG_PATH` は `infer.py` からも使う

#### `crane_pp/infer.py` — 学習済みポリシーで動かすノード
学習した SmolVLA で行動を推論し、アームに送る。torch が必要なので `ros2 run` ではなく `pixi run infer`（infer 環境の Python）で起動する。
1. `checkpoints/smolvla_ckpt.pt` を読み込む。VLM（SmolVLM2）は Hugging Face から読み、学習した部分（action head など）だけを ckpt から戻す
2. 最初に、ブロックを初期位置に戻し、アームを初期姿勢にする（デモと同じ初期条件）
3. 最新の俯瞰画像・関節角・タスク文から、10 ステップ（1 秒分）の関節角を推論する
4. 10 ステップを 1 本の軌道としてアームとグリッパに送り、実行し終わったら 3 に戻る（約 26 回の繰り返しで置き場まで運ぶ）

推論を安定させるための工夫:
- 初期ノイズを 0 に固定する（ランダムだと推論ごとに結果が大きくぶれた）
- 指令をデモの関節角の範囲に収める（`np.clip`。範囲外に行くと学習データにない状態になって止まった）
- 画像の受信待ちの列を 1 枚にし、前の軌道を実行し終わった後の画像・関節角がそろうまで待つ（古い画像で推論すると、違う段階の動きを交互に出した）

#### `package.xml` / `setup.py` / `setup.cfg` / `resource/crane_pp` / `crane_pp/__init__.py` — パッケージの定型
ament_python パッケージに必要な決まりのファイル。`setup.py` で launch と world をインストール先にコピーし、`ros2 run crane_pp demo` のコマンドを登録している。

### 学習まわり
#### `scripts/rosbag2lerobot.py` — rosbag → LeRobotDataset の変換
講義の `pai_training` の同名スクリプトと同じ流れ。
1. `data/raw/` の各 bag（1 bag = 1 エピソード）から、`/joint_states`・アーム指令・グリッパ指令・`/image_raw` を読む
2. 4 トピックがそろっている時間を 0.1 秒（10fps）刻みに区切り、各時刻で「その時刻以前の一番新しい値」を使う（直前値保持）。トピックごとに周期がばらばらなのをそろえるため
3. 1 フレーム = 画像 `observation.images.top`・関節角 `observation.state`（5 次元）・指令 `action`（5 次元）・タスク文、として LeRobotDataset に追加する
- 結果は 25 エピソード・7150 フレーム（1 エピソード約 285 フレーム）。画像は mp4 動画として保存される

#### `model.py` — SmolVLA + Flow Matching のモデル
講義ノートブック `reference/basic_final_assignment.ipynb` の 8 章を切り出したもの。学習（Colab）と推論（`infer.py`）の両方で import する。
- `SinusoidalTimeEmbedding`: Flow Matching の時刻 t（0〜1）を sin/cos のベクトルにする
- `TransformerBlock`: self-attention（10 ステップの行動どうし）→ cross-attention（画像・文・関節角を参照）→ MLP の 1 層
- `FlowMatchingHead`: TransformerBlock を 4 層重ね、「ノイズから行動へ向かう速度」を予測する
- `RealVLMEncoder` / `load_vlm_encoder()`: SmolVLM2-500M を読み込んで凍結し、画像とタスク文を特徴量にする。画像のタイル分割はしない（`do_image_splitting=False`。トークン数を抑えて GPU メモリ不足を防ぐ）
- `SmolVLA`: 全体をまとめるクラス
  - `compute_loss()`（学習）: ノイズ x0 と正解の行動 a の間の点 x(t) = (1 − t) × x0 + t × a で、予測した速度と a − x0 の二乗誤差をとる
  - `sample_actions()`（推論）: x0 から始め、x ← x + vθ(x, t, 条件) × 0.1 を 10 回くり返して行動を作る（Euler 法）
- 学習するのは action head・VLM 出力の射影層・関節角の射影層だけで、VLM は凍結したまま

#### `notebooks/train.ipynb` — Colab での学習
講義ノートブックの 9 章（学習ループ・正規化）をもとに、データを自前の LeRobotDataset に差し替えたもの。章立て:
1. GPU（T4）の確認
2. LeRobot v0.6.0 のインストール（変換と同じ版）
3. `crane_pp_dataset.zip` と `model.py` のアップロード・解凍
4. SmolVLA の組み立て（VLM は凍結、action head は 4 層・hidden 256）
5. データセットの読み込み（各フレームに「その先 10 ステップ分の action」を付ける）
6. action の正規化（LeRobot が変換時に計算した mean / std を使う）
7. 数ステップ回して 1 ステップの時間を測り、25 分に収まるステップ数を決める
8. 学習（AdamW、lr 3e-4、warmup → cosine、batch 8。25 分で打ち切り）
9. loss のグラフ表示と、`smolvla_ckpt.pt` の保存・ダウンロード

### その他
- `README.md`（このファイル）
- `report/report.html` → `report/report.pdf`: 提出用レポート（HTML で書き、Chrome で PDF にした）
- `video/pick_place_2x.mp4`: 提出動画（2 倍速）
- `reference/`: 講義資料・参考ノートブック（git 管理外。読むだけ）

## 実行方法

### 前提
- Docker Desktop（Mac の場合は Apple Silicon でも動く。イメージは amd64 なのでエミュレーションで動作する）
- Google アカウント（Colab で学習するため）
- コマンドの実行場所は 3 か所ある。**`docker ...` は Mac のターミナル**、`pixi run sim` など画面を出すものは **noVNC のターミナル**、学習は **Colab**

### 0. コンテナの準備（初回のみ）
```bash
# Mac のターミナル（プロジェクトのディレクトリで）。プロジェクトをコンテナの ~/work にマウントする
docker run -d --name crane --platform linux/amd64 -p 6080:80 --shm-size=512m \
  -v "$PWD":/home/ubuntu/work airobotbook/ros2-desktop-ai-robot-book-humble:latest
# 2 回目以降は docker start crane

# コンテナに pixi を入れ、自作パッケージをビルドする
docker exec -u ubuntu crane bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
docker exec -u ubuntu -w /home/ubuntu/work crane /home/ubuntu/.pixi/bin/pixi run build
```
ブラウザで http://localhost:6080/vnc.html を開く（パスワード `ubuntu`）。

以下、Mac のターミナルからのコマンドは `docker exec -u ubuntu -w /home/ubuntu/work crane /home/ubuntu/.pixi/bin/pixi run <タスク>` の形で実行する（下では `pixi run <タスク>` と略す）。

### 1. シミュレーションを起動する（noVNC のターミナル）
```bash
cd ~/work && pixi run sim
```
Gazebo と RViz が開き、テーブル上にアーム・赤ブロック・緑の置き場が表示される。

### 2. デモを収集する（Mac のターミナル、Gazebo は起動したまま）
```bash
pixi run demo     # お試しで 1 回だけデモを流す（記録しない）
pixi run record   # デモ 25 回を data/raw/ep0〜ep24 に記録する（約 15 分）
```

### 3. データを変換する（Mac のターミナル）
```bash
pixi run convert      # data/raw → data/interim/crane_pp（LeRobotDataset）
pixi run zip-dataset  # Colab にアップロードする crane_pp_dataset.zip を作る
```

### 4. 学習する（Colab）
1. Colab で `notebooks/train.ipynb` を開き、ランタイムを T4 GPU にする
2. 上から順に実行し、3 章で `crane_pp_dataset.zip` と `model.py` をアップロードする
3. 最後にダウンロードされる `smolvla_ckpt.pt` を `checkpoints/smolvla_ckpt.pt` に置く

### 5. 推論する（Mac のターミナル、Gazebo は起動したまま）
```bash
pixi run infer                         # 学習済みポリシーでアームを動かす
docker exec crane pkill -f infer.py    # 止めるとき
```
1 回の推論に約 9 秒かかるため、アームは「1 秒動いて約 9 秒止まる」を繰り返す。約 4 分でブロックを置き場に置く。

### 6. 動画を作る
Mac の画面収録（Command + Shift + 5）で noVNC の画面を撮り、`video/raw.mov` に置いてから:
```bash
pixi run video   # video/pick_place_2x.mp4（2 倍速、上部に赤字「2倍速再生」）を作る
```

## pixi のタスク一覧
| タスク | 実行場所 | 内容 |
|---|---|---|
| `build` | コンテナ | 自作パッケージ crane_pp を colcon でビルド |
| `sim` | noVNC | Gazebo + RViz を起動 |
| `demo` | コンテナ | デモの動きを 1 回流す |
| `record` | コンテナ | デモ 25 回を rosbag に記録 |
| `convert` | コンテナ | rosbag → LeRobotDataset |
| `zip-dataset` | コンテナ | Colab 用に zip にまとめる |
| `infer` | コンテナ | 学習済みポリシーで推論 |
| `video` | コンテナ | 画面収録を 2 倍速の mp4 にする |

## 主な工夫
- デモの関節角を、腕の長さから計算して決めた（調整なしでつかめた）
- 学習の前に 1 ステップの時間を測り、25 分で終わるステップ数を自動で決めた
- VLM の画像のタイル分割をやめて、メモリ不足を解消し推論も速くした
- 推論を安定させるため、初期ノイズの固定・指令範囲の制限・最新画像での推論を行った
- Python の版の違い（LeRobot は 3.12 以上、ROS 2 Humble は 3.10）に、pixi の環境を 2 つに分けて対応した