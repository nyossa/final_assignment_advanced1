from glob import glob

from setuptools import setup

package_name = 'crane_pp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ament にパッケージを認識させるための定型
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ros2 launch crane_pp sim.launch.py で使う launch と world をインストールする
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nyossa',
    maintainer_email='nyossa@example.com',
    description='CRANE+ V2 pick and place with SmolVLA (Flow Matching) on Gazebo',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # ros2 run crane_pp demo でデモ軌道を流す
            # （infer.py は torch が必要なため ros2 run ではなく pixi run infer で起動する）
            'demo = crane_pp.demo:main',
        ],
    },
)
