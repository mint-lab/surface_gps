from setuptools import setup
from glob import glob
import os

package_name = 'surface_gps'
SHARE_DIR = os.path.join("share", package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join(SHARE_DIR, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join(SHARE_DIR, "config"), glob(os.path.join("config", "*.urdf"))),
        (os.path.join(SHARE_DIR, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyunkil',
    maintainer_email='hyunkil@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'ekf_node = surface_gps.ekf_node:main',
        ],
    },
)
