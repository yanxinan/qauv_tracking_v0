from setuptools import find_packages, setup

package_name = 'data_collection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
            'mujoco_log_node = data_collection_pkg.mujoco_log_node:main',
            'data_log_node = data_collection_pkg.data_log_node:main',
        ],
    },
)
