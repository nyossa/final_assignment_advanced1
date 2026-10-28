# ==============================================================
# rosbag → LeRobotDataset 変換
#   講義（session5）の matsuolab/pai_training の scripts/rosbag2lerobot.py と同じ流れ:
#     1. data/raw/ 以下の各 bag（1 bag = 1 エピソード）から必要なトピックを読む
#     2. 全トピックの共通時間範囲を fps 間隔でサンプリングし、各時刻の「直前の値」を使う
#     3. アームとグリッパの指令を結合して action にし、LeRobotDataset に 1 フレームずつ追加
#   pai_training 版からの変更点:
#     - 俯瞰カメラ画像 observation.images.top を追加（VLA の入力）
#     - state / action を関節名で並べ替える（/joint_states の並び順に依存しない）
#     - エピソード長を最長にそろえる処理は省略（デモ軌道は全エピソード同じ長さのため）
#     - LeRobot v0.6.0（Colab の学習ノートブックと同じ版）に合わせて import と finalize() を変更
# ==============================================================
import shutil
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG_DIR = Path('data/raw')  # ros2 bag record の出力先（ep0, ep1, ...）
OUTPUT_DIR = Path('data/interim/crane_pp')  # 変換後の LeRobotDataset
REPO_ID = 'crane_pp'  # データセット名（ローカル保存のみで Hub には上げない）
IMAGE_SIZE = 256  # pick_place.sdf のカメラ解像度（256x256）

cfg = yaml.safe_load(open('config/sim.yaml'))
JOINT_NAMES = cfg['arm_joints'] + [cfg['gripper_joint']]

# 5 次元 [joint1..joint4, gripper] の state / action と、俯瞰カメラ画像（mp4 で保存される）
FEATURES = {
    'observation.state': {'dtype': 'float32', 'shape': (5,), 'names': ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']},
    'action': {'dtype': 'float32', 'shape': (5,), 'names': ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']},
    'observation.images.top': {'dtype': 'video', 'shape': (IMAGE_SIZE, IMAGE_SIZE, 3), 'names': ['height', 'width', 'channels']},
}

# rosbag のメッセージを Python オブジェクトに戻すための型定義（ROS2 Humble）
typestore = get_typestore(Stores.ROS2_HUMBLE)


def read_messages(bag_path):
    """bag から必要な 4 トピックの (受信時刻[ns], 値) を取り出す."""
    topics = [cfg['joint_states_topic'], cfg['arm_topic'], cfg['gripper_topic'], cfg['image_topic']]
    data = {t: ([], []) for t in topics}
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == cfg['joint_states_topic']:
                # 関節名 → 角度 の辞書から JOINT_NAMES の順で 5 次元にする
                pos = dict(zip(msg.name, msg.position))
                value = [pos[n] for n in JOINT_NAMES]
            elif connection.topic == cfg['image_topic']:
                # sensor_msgs/Image (rgb8) を (H, W, 3) の uint8 配列にする
                value = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                # JointTrajectory の 1 点目を指令値として使う（demo.py は 1 点だけ送る）
                value = list(msg.points[0].positions)
            data[connection.topic][0].append(timestamp)
            data[connection.topic][1].append(value)
    return {t: (np.array(ts), np.array(vs)) for t, (ts, vs) in data.items()}


def sample_and_hold(times, values, target_times):
    """各 target_time について、その時刻以前で一番新しい値を使う（直前値保持）."""
    idx = np.searchsorted(times, target_times, side='right') - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def main():
    # 作り直せるように、既存の変換結果は消してから作成する
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=cfg['fps'], root=OUTPUT_DIR,
        robot_type='crane_plus', features=FEATURES, use_videos=True)

    # metadata.yaml を持つディレクトリ = 1 エピソードの bag
    bag_paths = sorted(p.parent for p in BAG_DIR.glob('*/metadata.yaml'))
    print(f'Found {len(bag_paths)} episodes')
    for ep, bag_path in enumerate(bag_paths):
        msgs = read_messages(bag_path)

        # 全トピックがそろっている時間範囲を fps 間隔で区切る
        start = max(ts[0] for ts, _ in msgs.values())
        end = min(ts[-1] for ts, _ in msgs.values())
        target_times = np.arange(start, end, 1e9 / cfg['fps'])

        # 各トピックを直前値保持でサンプリング
        state = sample_and_hold(*msgs[cfg['joint_states_topic']], target_times)
        arm = sample_and_hold(*msgs[cfg['arm_topic']], target_times)
        gripper = sample_and_hold(*msgs[cfg['gripper_topic']], target_times)
        image = sample_and_hold(*msgs[cfg['image_topic']], target_times)
        action = np.hstack([arm, gripper])  # [アーム4軸 + グリッパ1軸]

        for i in range(len(target_times)):
            dataset.add_frame({
                'observation.state': state[i].astype(np.float32),
                'action': action[i].astype(np.float32),
                'observation.images.top': image[i],
                'task': cfg['task'],
            })
        dataset.save_episode()
        print(f'episode {ep}: {len(target_times)} frames from {bag_path}')

    # parquet のフッタ等を書き込んでデータセットを完成させる（LeRobot v3 形式では必須）
    dataset.finalize()
    print(f'Dataset saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
