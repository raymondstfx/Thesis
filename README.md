# Thesis
Honours Thesis

## Project Description
This project references the implementation of [MobileCount](https://github.com/ChenyuGAO-CS/MobileCount) and modifies upon it.

## Virtual Environment
Install [pyenv-win](https://github.com/pyenv-win/pyenv-win) first, then create a new virtual environment:
```bash
python -m venv .venv
```
Activate:
```bash
.\.venv\Scripts\Activate
```

## Dependencies
### Cuda (Using GeForce GTX 1080 Ti)
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Dependencies
Update setuptools:
```bash
pip install --upgrade setuptools
```
Install dependencies:
```bash
pip install certifi==2024.12.14 charset-normalizer==3.4.1 contourpy==1.3.1 cycler==0.12.1 fonttools==4.55.3 h5py==3.12.1 idna==3.10 kiwisolver==1.4.8 matplotlib==3.10.0 numpy==1.26.4 opencv-python==4.10.0.84 packaging==24.2 pillow==11.1.0 pip==23.0.1 pyparsing==3.2.1 python-dateutil==2.9.0.post0 requests==2.32.3 setuptools==65.5.0 six==1.17.0 typing_extensions==4.12.0 urllib3==2.3.0
```

## Run
First, training is performed, and then a weight file (i.e., a .pth file) will be generated:
```bash
python train.py
```

To generate a density map, run:
```bash
python visual.py
```
It will be generated in the `Visual` folder.
