# conda create -n fpn
# conda activate fpn

read -p "Did you activate the environment (y/n)?" CONT
if [ "$CONT" != "y" ]; then
    exit 1
fi

sudo apt install gcc g++
conda install python -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip3 install tqdm
pip install git+https://github.com/cocodataset/panopticapi.git
python detectron2_install.py
