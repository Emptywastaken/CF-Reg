# COCE Repository

![Python 3.9 Supported](https://img.shields.io/badge/Python-3.9%20Supported-black.svg?style=flat&logo=python&color=gold&labelColor=black)


**Required packages: carla-recourse-3.9, check the folder external_libs**

To use the repo create a conda env using the following command 
```
conda create --name coce python=3.9
```

Once the env has been created use:

```
conda activate coce
```

Then install carla-recourse-3.9 using:

```
python -m pip install external_libs/carla_recourse-1.0.0-py39-none-any.whl
```