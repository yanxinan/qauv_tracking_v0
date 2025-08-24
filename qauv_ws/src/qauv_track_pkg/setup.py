from setuptools import find_packages, setup

package_name = 'qauv_track_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    py_modules=['mypid', 'my_normalization'],
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
            'track_pid_omega_node = qauv_track_pkg.track_pid_omega_node:main',
            'track_pid_angle_node = qauv_track_pkg.track_pid_angle_node:main',
            'track_pid_position_node = qauv_track_pkg.track_pid_position_node:main',
            'track_pid_posiome_node = qauv_track_pkg.track_pid_posiome_node:main',
            'track_bc_position_node = qauv_track_pkg.track_bc_position_node:main',
            'data_log_node = qauv_track_pkg.data_log_node:main',
        ],
    },
)
