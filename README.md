# Project Pore-Net

<img width="444" alt="image" src="https://github.com/user-attachments/assets/fb7285bc-0b0d-4836-b160-67558601113c" />

Learn Pore Scale Physics in the way of Machine Learning

## Setup:

1. Conda env configuration:

```shell
conda env create -f environment.yml
```

+ if you wish to update environment dependencies (not recommended) please do:

```shell
conda env update --name pore_net --file environment.yml --prune
```

+ if you wish to register this kernel to jupyterlab:

```shell
python -m ipykernel install --user --name=pore_net --display-name "pore_net"
```


1. Package setup
```
pip install -e <path-to-folder-root>
```