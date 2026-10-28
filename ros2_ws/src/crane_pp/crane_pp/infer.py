# ==============================================================
# 学習済み SmolVLA（Flow Matching）で推論し、関節指令を publish するノード
#   配線は講義の matsuolab/pai_training の scripts/infer.py と同じ:
#     /joint_states を subscribe → 方策で action を推論 → アーム・グリッパの指令トピックに publish
#   変更点:
#     - 方策を ACT から自作の SmolVLA（model.py）に置き換え、俯瞰カメラ画像も入力する
#     - 1 回の推論で chunk_size 個（1 秒ぶん）の action を 1 本の軌道として送り、
#       実行し終わってから次の推論をする（CPU 推論が遅いので毎ステップ推論はしない）
#   起動は pixi run infer（torch が必要なため ros2 run ではなく pixi の Python で動かす）
# ==============================================================
import argparse

import numpy as np
import rclpy
import torch
import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from crane_pp.demo import CONFIG_PATH, reset_block
from model import SmolVLA, load_vlm_encoder


def make_trajectory(joint_names, positions, dt):
    """(K, n_joints) の目標角列を、dt 秒刻みで順に通る JointTrajectory にする."""
    msg = JointTrajectory()
    msg.joint_names = joint_names
    for k, pos in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in pos]
        t = (k + 1) * dt  # k 番目の点に到達する時刻（最初の点は dt 秒後）
        pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1) * 1e9))
        msg.points.append(pt)
    return msg


class InferenceNode(Node):
    def __init__(self, ckpt_path):
        # Gazebo の時刻で動かす（demo.py と同じ。chunk の実行待ちもシム時間で数える）
        super().__init__('crane_pp_infer', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.cfg = yaml.safe_load(open(CONFIG_PATH))
        self.joint_names = self.cfg['arm_joints'] + [self.cfg['gripper_joint']]
        self.dt = 1.0 / self.cfg['fps']  # action 1 step の時間（学習データの fps と同じ）
        # 指令の上下限 = デモのウェイポイントで使った関節角の範囲（学習データの範囲とほぼ同じ）。
        # 範囲外（例: joint1 = +2.15）に行き過ぎると、学習データに無い状態になって次の段階へ進めなくなるため
        wps = np.array(self.cfg['waypoints'])[:, :5]
        self.action_min = wps.min(axis=0)
        self.action_max = wps.max(axis=0)

        # --- 学習済みモデルの読み込み（CPU, float32） ---
        # VLM は Hugging Face から読み、学習した部分（flow_head / vlm_projector / state_proj）は ckpt から戻す
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)  # tensor と数値だけなので安全な読み方にする
        self.policy = SmolVLA(load_vlm_encoder('cpu', torch.float32),
                              action_dim=ckpt['action_dim'], state_dim=ckpt['state_dim'],
                              chunk_size=ckpt['chunk_size'], transformer_hidden_dim=ckpt['hidden_dim'])
        self.policy.flow_head.load_state_dict(ckpt['flow_head_state_dict'])
        self.policy.vlm_projector.load_state_dict(ckpt['vlm_projector_state_dict'])
        self.policy.state_proj.load_state_dict(ckpt['state_proj_state_dict'])
        self.policy.eval()
        # 学習時の action 正規化の統計量（推論結果を実スケールに戻すのに使う）
        self.action_mean = ckpt['action_mean']
        self.action_std = ckpt['action_std']

        # --- トピック ---
        self.arm_pub = self.create_publisher(JointTrajectory, self.cfg['arm_topic'], 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, self.cfg['gripper_topic'], 10)
        self.create_subscription(JointState, self.cfg['joint_states_topic'], self.joint_state_callback, 10)
        # 画像は受信待ちの列を 1 枚にして、常に最新の 1 枚だけを持つ
        # （エミュレーション上の Python では画像の受信が追いつかず、列が詰まると数秒〜十数秒前の画像で推論してしまうため）
        self.create_subscription(Image, self.cfg['image_topic'], self.image_callback, 1)
        self.state = None  # 最新の関節状態 (5,)
        self.image = None  # 最新の俯瞰カメラ画像 (H, W, 3)
        self.state_time = None  # その関節状態のシム時刻（メッセージの header.stamp）
        self.image_time = None  # その画像のシム時刻
        self.next_time = None  # 次に推論してよいシム時刻（None = まだ初期化していない）

        # 0.1 秒ごとに「次の推論をしてよいか」を確認する
        self.create_timer(0.1, self.step)

    def reset_episode(self):
        """エピソードの初期条件をデモ収集と同じにする.

        ブロックを初期位置に戻し、ロボットを初期姿勢（waypoints の 1 行目）に 2 秒かけて戻す。
        /clock の受信とトピックの接続を待つため、__init__ ではなく最初のタイマー呼び出しで行う。
        """
        reset_block(self.cfg['block_pose'])
        init = np.array(self.cfg['waypoints'][0][:5])
        self.arm_pub.publish(make_trajectory(self.cfg['arm_joints'], [init[:4]], 2.0))
        self.gripper_pub.publish(make_trajectory([self.cfg['gripper_joint']], [init[4:]], 2.0))
        self.next_time = self.get_clock().now() + Duration(seconds=3.0)  # 初期姿勢に着くまで推論を待つ

    def joint_state_callback(self, msg):
        # 関節名で並べ替えて [joint1..joint4, gripper] にする（rosbag2lerobot.py と同じ）
        pos = dict(zip(msg.name, msg.position))
        self.state = np.array([pos[n] for n in self.joint_names], dtype=np.float32)
        self.state_time = Time.from_msg(msg.header.stamp)

    def image_callback(self, msg):
        # sensor_msgs/Image (rgb8) → (H, W, 3) uint8（rosbag2lerobot.py と同じ）
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.image_time = Time.from_msg(msg.header.stamp)

    def step(self):
        # 最初の 1 回だけエピソードを初期化する
        if self.next_time is None:
            self.reset_episode()
            return
        # 前の chunk を実行し終わった「後」に撮られた画像と関節状態がそろうまで待つ（古い観測で推論しないため）
        if self.state is None or self.image is None:
            return
        if self.image_time < self.next_time or self.state_time < self.next_time:
            return

        # 現在の画像・state・タスク文から action chunk を推論し、正規化を戻す
        states = torch.from_numpy(self.state).unsqueeze(0)  # (1, 5)
        # 初期ノイズを 0 に固定する（ランダムだと推論ごとに段階が前後に飛び、ブロックの真上に収束しなかったため）
        actions = self.policy.sample_actions([self.image], [self.cfg['task']], states, noise_scale=0.0)  # (1, K, 5)
        actions = (actions[0] * self.action_std + self.action_mean).numpy()  # (K, 5)
        actions = np.clip(actions, self.action_min, self.action_max)  # デモの関節角の範囲に収める

        # chunk 全体を 1 本の軌道として送る（アーム 4 軸とグリッパ 1 軸で別トピック）
        self.arm_pub.publish(make_trajectory(self.cfg['arm_joints'], actions[:, :4], self.dt))
        self.gripper_pub.publish(make_trajectory([self.cfg['gripper_joint']], actions[:, 4:], self.dt))
        self.get_logger().info(f'action[0]: {actions[0].round(3)}  action[-1]: {actions[-1].round(3)}')

        # chunk を実行し終わる時刻まで次の推論を待つ
        self.next_time = self.get_clock().now() + Duration(seconds=len(actions) * self.dt)


def main():
    parser = argparse.ArgumentParser(description='SmolVLA inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--ckpt', default='checkpoints/smolvla_ckpt.pt', help='train.ipynb が保存したチェックポイント')
    args = parser.parse_args()
    rclpy.init()
    node = InferenceNode(args.ckpt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
