# ==============================================================
# シミュレーション起動（Gazebo + RViz + カメラ画像のブリッジ）
#   イメージ内の crane_plus_gazebo/launch/crane_plus_with_table.launch.py（2024-09 版）を複製したもの。
#   その版は world が table.sdf に固定で、カメラ画像のブリッジも無いため、include ではなく複製して
#   次の 2 点だけを変えている（それ以外は元ファイルと同じ）:
#     1. world を自作の pick_place.sdf（赤ブロック・緑の置き場・俯瞰カメラ入り）にする
#     2. ros_gz_bridge で俯瞰カメラの画像 image_raw も ROS の /image_raw に中継する
# ==============================================================
import os

from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_launch_description():
    # PATHを追加で通さないとSTLファイルが読み込まれない（元ファイルのコメントのまま）
    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ['LD_LIBRARY_PATH'],
           'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(
               get_package_share_directory('crane_plus_description'))}
    # 【変更 1】自作の world を使う
    world_file = os.path.join(
        get_package_share_directory('crane_pp'), 'worlds', 'pick_place.sdf')
    gui_config = os.path.join(
        get_package_share_directory('crane_plus_gazebo'), 'gui', 'gui.config')
    # -r オプションで起動時にシミュレーションをスタートしないと、コントローラが起動しない
    ign_gazebo = ExecuteProcess(
            cmd=['ign gazebo -r', world_file, '--gui-config', gui_config],
            output='screen',
            additional_env=env,
            shell=True
        )

    # ロボット（/robot_description）をテーブル天板の高さ 1.015m に生成する
    gazebo_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-name', 'crane_plus',
                   '-z', '1.015',
                   '-allow_renaming', 'true'],
    )

    # Gazebo（Ignition）用のロボットモデルと ros2_control 設定を読み込む
    description_loader = RobotDescriptionLoader()
    description_loader.use_gazebo = 'true'
    description_loader.use_ignition = 'true'
    description_loader.gz_control_config_package = 'crane_plus_control'
    description_loader.gz_control_config_file_path = 'config/crane_plus_controllers.yaml'
    description = description_loader.load()

    # MoveIt2 と RViz（動画で Gazebo と並べて映す）
    move_group = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('crane_plus_moveit_config'),
                '/launch/run_move_group.launch.py']),
            launch_arguments={'loaded_description': description}.items()
        )

    # 関節状態の配信（/joint_states）とアーム・グリッパのコントローラを起動する
    spawn_joint_state_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner joint_state_broadcaster'],
                shell=True,
                output='screen',
            )

    spawn_arm_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner crane_plus_arm_controller'],
                shell=True,
                output='screen',
            )

    spawn_gripper_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner crane_plus_gripper_controller'],
                shell=True,
                output='screen',
            )

    # Gazebo → ROS のトピック中継
    #   /clock: シム時刻（元ファイルと同じ）
    #   【変更 2】image_raw: 俯瞰カメラ画像（pick_place.sdf のカメラの topic）。
    #   書き方は GitHub の新しい版の crane_plus_with_table.launch.py と同じ
    bridge = Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                           'image_raw@sensor_msgs/msg/Image[ignition.msgs.Image'],
                output='screen'
            )

    return LaunchDescription([
        # Gazebo の時刻（/clock）で全ノードを動かす
        SetParameter(name='use_sim_time', value=True),
        ign_gazebo,
        gazebo_spawn_entity,
        move_group,
        spawn_joint_state_controller,
        spawn_arm_controller,
        spawn_gripper_controller,
        bridge
    ])
