<h1 align="center">Unified Action Recognition System</h1>

Presenting a highly customizable action recognition system that enables ease of adapting to any real world application.

This system can be fine-tuned to recgonize as many actions as needed, by just  providing a few videos for each action. The resulting fine-tuned model is capable of real-time action recognition with webcam.

<h2 align="center">Demo</h2>
    <p align="center">
    Fall Detection
    <img src="./assets/fall.gif" width=100% class="center">
    Ball Kicking Analysis
    <img src="./assets/kick.gif" width=100% class="center">
    Aircraft Marshaller Signals
    <img src="./assets/guide.gif" width=100% class="center">
    </p>

<h2 align="center">Setup Environment for Demo</h2>
<p align="left">

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
</p>

<h3 align="center">Clone the repository locally</h3>

```
git clone https://github.com/kaikaic1998/Unified_Action_Recognition_System.git
cd Unified_Action_Recognition_System
```
<h3 align="center">Installation</h3>

**Step 1.** Install libraries
```
pip install -r requirements.txt
```

**Step 2.** Install Cython_bbox
```
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```
**Step 3.** Install lap
```
pip install lap
```
If above is not successful, try below:
```
git clone https://github.com/gatagat/lap.git
cd lap
python setup.py build
python setup.py install
cd ../
```
**Step 4.** Install Pytorch

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
