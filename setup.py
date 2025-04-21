from setuptools import find_packages, setup

package_name = 'BRAVO_AGV'

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
    maintainer='ricbear',
    maintainer_email='ricberar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bravo_node = BRAVO_AGV.bravo_node:main'
	    'orbbec_det = BRAVO_AGV.ros_color_depth:main'
        ],
    },
)
