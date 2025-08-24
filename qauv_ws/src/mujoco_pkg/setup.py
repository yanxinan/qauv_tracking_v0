from setuptools import find_packages, setup

package_name = 'mujoco_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    py_modules=['mytrajectory'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zb',
    maintainer_email='zb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_simulator_node = mujoco_pkg.mujoco_simulator_node:main',
            'mujoco_grasp_node = mujoco_pkg.mujoco_grasp_node:main',
            'mujoco_track_node = mujoco_pkg.mujoco_track_node:main',
        ],
    },
)
