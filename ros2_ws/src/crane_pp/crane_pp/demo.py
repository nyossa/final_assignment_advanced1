# ==============================================================
# デモ軌道を publish するノード（学習データ収集用）
#   config/sim.yaml の固定ウェイポイントを線形補間し、physicalai_teleop の
#   teleop_keyboard.py と同じく 20Hz でアーム・グリッパへ関節指令を送る。
#   ros2 bag record と同時に動かして 1 エピソード = 1 bag を記録する。
# ==============================================================
import subprocess

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# コンテナ内のプロジェクトのマウント先（CLAUDE.md の docker run を参照）
CONFIG_PATH = '/home/ubuntu/work/config/sim.yaml'
RATE = 20.0  # 指令の周波数 [Hz]（teleop_keyboard.py と同じ）


def reset_block(block_pose):
    """Gazebo の set_pose サービスで赤ブロックを初期位置に戻す（エピソードの初期条件をそろえる）."""
    x, y, z = block_pose
    subprocess.run(
        ['ign', 'service', '-s', '/world/default/set_pose',
         '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
         '--timeout', '3000',
         '--req', f'name: "red_block", position: {{x: {x}, y: {y}, z: {z}}}'],
        check=True)


def make_command(joint_names, positions):
    """関節名と目標角から、0.1 秒後に到達する 1 点だけの JointTrajectory を作る（teleop_keyboard.py と同じ形）."""
    msg = JointTrajectory()
    msg.joint_names = joint_names
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in positions]
    pt.time_from_start.nanosec = int(0.1 * 1e9)
    msg.points.append(pt)
    return msg


class DemoNode(Node):
    def __init__(self):
        # Gazebo の時刻で動かす（シムが実時間より遅くても軌道の速さが変わらないように）
        super().__init__('crane_pp_demo',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.cfg = yaml.safe_load(open(CONFIG_PATH))
        self.arm_pub = self.create_publisher(JointTrajectory, self.cfg['arm_topic'], 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, self.cfg['gripper_topic'], 10)

        # ウェイポイント [j1..j4, gripper, 秒] を 20Hz の指令列 (N, 5) に展開する
        wps = np.array(self.cfg['waypoints'])
        commands = []
        prev = wps[0, :5]
        for wp in wps:
            n = int(round(wp[5] * RATE))
            for k in range(1, n + 1):
                commands.append(prev + (wp[:5] - prev) * k / n)  # 直前のウェイポイントから線形補間
            prev = wp[:5]
        self.commands = commands
        self.step = 0
        self.done = False

        # エピソード開始前にブロックを初期位置へ戻す
        reset_block(self.cfg['block_pose'])
        self.timer = self.create_timer(1.0 / RATE, self.publish_command)
        self.get_logger().info(f'demo start: {len(commands)} commands ({len(commands) / RATE:.1f} s)')

    def publish_command(self):
        # 全指令を送り終えたら終了フラグを立てる（main のループが抜ける）
        if self.step >= len(self.commands):
            self.done = True
            return
        cmd = self.commands[self.step]
        self.arm_pub.publish(make_command(self.cfg['arm_joints'], cmd[:4]))
        self.gripper_pub.publish(make_command([self.cfg['gripper_joint']], cmd[4:]))
        self.step += 1


def main():
    rclpy.init()
    node = DemoNode()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)
    node.get_logger().info('demo done')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
