### Step 1: Installing the Deep Learning Framework and Dependencies

#### For TensorFlow:

```bash
# Create a virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # Activate the virtual environment

# Install TensorFlow
pip install tensorflow
```

#### For Pytorch

```bash
# Create a virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # Activate the virtual environment

# Install PyTorch (with CUDA support if available)
# Replace 'torch' and 'torchvision' versions with appropriate ones for your CUDA version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

```
### Setting Up IDE/Development Environment
**Jupyter Notebook:**
1.Install Jupyter Notebook (if not installed):
```bash

pip install notebook
```
2.Launch Jupyter notebook
```bash
jupyter notebook
```

**Pycharm or Any IDE**

- Download and install Pycharm from JetBrains Website

- Create a new projects in Pycharm

- Configure the Python interpreter in Pycharm to use the virtual environment created earlier

#### **Notes**
- Replace *myenv*  with ypur perfered virtual enviroment name 

- Ensure that the chosen IDE or text editor supports the selected framework and allows for easy dataset loading and manipulation. 
