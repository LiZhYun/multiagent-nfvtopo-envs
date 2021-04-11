from setuptools import setup, find_packages

setup(name='multiagentnfv',
      version='0.0.1',
      description='Multi-Agent Goal-Driven NFVTopo Environment',
      # url='https://github.com/openai/multiagent-public',
      author='ZhiYuan Li',
      author_email='zhiyuanli@std.uestc.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
