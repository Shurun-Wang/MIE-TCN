# MIE-TCN for Lower-extremity motion estimation using one IMU sensor 
![Visitors](https://api.visitorbadge.io/api/visitors?path=https://github.com/Shurun-Wang/MIE-TCN&label=visitors&countColor=%232ccce4&style=plastic)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org/)

Welcome to the official repository for the MIE-TCN method.


---

## 📑 Table of Contents
1. [Data Preparation](#-data-preparation)
2. [Quick Start](#-quick-start)
3. [Repository Structure](#-repository-structure)
4. [License](#-license)

---

-----

## 📊 Data Preparation

This project involves the processing of multiple physiological datasets. To ensure full reproducibility, we provide comprehensive preprocessing guidelines.

### 1\. Download Datasets

Please download the raw EEG datasets from their official sources and place them in the `data/[dataset_name]` directory. The framework currently supports:

  * **database1**: A multi-sensor human gait dataset captured through an optical system and inertial measurement units.
  * **database2**: A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions.
  * **database3**: The COMPWALK-ACL: A dataset of multi-pace IMU gait kinematics in adolescents, adults, and acl injured patients.
 
### 2\. Run Preprocessing

Taking the preprocessing of the database1 as an example:
```bash
python data/database1_process.py
```
The processed and segmented data will be automatically saved to the `data/database1_processed/` directory.


## 🚀 Quick Start

### MIE-TCN method

Execute the proposed MIE-TCN method:

```bash
python main_dl.py
```

## 📁 Repository Structure

```text
├── data/                   
│   ├── database1_processed/                       
│   ├── database2_processed/ 
│   ├── database3_processed/      
│   ├── data_loader_2.py    
│   ├── database1_process.py  
│   ├── database2_process.py  
│   ├── database3_process.py  
│   ├── features.py  
├── models/                 
├── utils/                
├── main_dl.py       
├── train_dl.py     
├── requirements.txt        
└── README.md                
```

-----

## 📜 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).


```
```

If our work is helpful to you, please **Star** it and kindly **Cite** our paper as:  
    
    @article{Wang2026EstimatingLM,
      title={Estimating Lower-Extremity Multi-Joint Kinematics with One IMU Sensor via Attention-based Temporal Convolutional Neural Network},
      author={Shurun Wang and Hao Tang and Ryutaro Himeno and Jordi Sol{\'e}-Casals and Ying Tan and Jie Mao and C{\'e}sar Federico Caiafa and Zhe Sun},
      journal={Cognitive Computation},
      year={2026},
      volume={18},
      url={https://api.semanticscholar.org/CorpusID:289984635}
    }
