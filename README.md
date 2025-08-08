COMP8539 – Assignment 1 🚀

## ✨ Project Overview

See Assignment1.pdf for details

---

## 📂 Repository Structure

├── task1\_1.py                 #Train a baseline model
├── requirements.txt           # Exact Python dependencies
├── runs/                      # Saved experiment logs & TensorBoard files
│   └── patch4\_dim256\_depth6\_heads8
│       ├── metrics.csv
│       └── plots/\*.png
└── README.md

## 🖥️ Quick-start

### 1. Clone & create a virtual-env

```bash
git clone https://github.com/Bornhug/COMP8539_Assignment1.git
cd COMP8539_Assignment1

conda create --name comp8539 python=3.8.20
conda activate comp8539
```

### 2. Install only the packages this project really needs

```bash
pip install -r requirements.txt  
```

### 3. Run the main experiment

1. **Dataset**:
   CIFAR-10
2. **Training**:
   This is the default setting (Also these hyperparameters gives best performance

```
python task1_1.py --patch_size 8 --dim 256 --depth 6 --heads 8 --epochs 100
```
