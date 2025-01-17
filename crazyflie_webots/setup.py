from setuptools import setup

package_name = 'crazyflie_webots'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
# data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py']))
data_files.append(('share/' + package_name + '/protos', ['protos/Crazyflie.proto']))
data_files.append(('share/' + package_name + '/meshes', [
    'meshes/ccw_prop.dae',
    'meshes/cf2_assembly.dae',
]))
data_files.append(('share/' + package_name + '/worlds', ['worlds/crazyflie.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/crazyflie.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joshua-ki',
    maintainer_email='joshuajdmk@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazyflie_driver = crazyflie_webots.crazyflie_driver:main'
        ],
    },
)
