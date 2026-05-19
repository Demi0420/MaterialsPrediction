# MaterialsPrediction

This repository contains the code, trained model weights, data files, generated structures, and additional experimental results associated with the revised manuscript.

The project mainly includes two models:

- `modelA.py`: a latent-space-based generative model for proposing new candidate material structures. It also supports model pretraining.
- `modelB.py`: a CGCNN-based property prediction model for predicting material properties such as melting point, density, and formation energy.

In the revised version, we additionally provide model variants, newly trained model weights, benchmark scripts, prediction scripts, and the 20 shortlisted pre-DFT CIF files corresponding to Table 2 in the revised manuscript.

---

## 1. Repository structure

The recommended repository structure is as follows:

```text
MaterialsPrediction/
├── README.md
├── requirements.txt
│
├── data_csv/
│   ├── data_all.csv
│   ├── data_all_with_volume.csv
│   └── data_e43V.csv
│
├── modelA.py
├── modelB.py
├── model_prototype.py
├── modelB-formula-group.py
├── predictB.py
├── benchmark_variants.py
├── test_cuda.py
│
├── modelA-weights/
│   ├── endecoder_model.pt
│   └── prototype_model.pth
│
├── modelB-weights/
│   ├── best_modelB_mp_finetuned.pth
│   ├── best_modelB_rho_finetuned.pth
│   └── best_modelB_mp_finetuned_grouped.pth
│
├── table2_shortlisted_pre_dft_cifs/
│   ├── 1_SrZr7Ge6.cif
│   ├── 2_RbSr(ZrGe)6.cif
│   └── ...
│
├── benchmark_outputs/
│   ├── full_guided/
│   ├── concat_only/
│   └── unconditional/
│
└── dft/
    └── to be added


## Requirements

Install the required dependencies using:

	```bash
	pip install -r requirements.txt

If GPU acceleration is used, please make sure that PyTorch is installed with the CUDA version compatible with the local environment.

CUDA availability can be checked by running:

	```bash
	python test_cuda.py


## Data files

The main data files are stored in the data_csv/ directory:

```text
data_csv/
├── data_all.csv
├── data_all_with_volume.csv
└── data_e43V.csv

Download links:

1. `data_all.csv` [Google Drive Link](https://drive.google.com/file/d/1iBU7PA1sMc4bHE1RyUUUm-0JBAhmv-Wh/view?usp=share_link).

2. `data_all_with_volume.csv` [Google Drive link](https://drive.google.com/file/d/1y8gZ2XQ4yVoF49rN4mG1BIXX0_f1PXN2/view?usp=share_link).

3. `data_e43V.csv` [Google Drive link](https://drive.google.com/file/d/1y8gZ2XQ4yVoF49rN4mG1BIXX0_f1PXN2/view?usp=share_link).


## Running Model A

Model A is used for predicting specific material properties. You can run the following commands:

### 1. Predict Melting Point

    python modelA.py --target_col melting_point_log --abrev mp

### 2. Predict Density

    python modelA.py --target_col density --abrev rho

### 3. Predict Formation Energy per Atom

    python modelA.py --target_col formation_energy_per_atom --abrev fe

💡 Ensure that the input CSV and model settings are correctly configured before running any prediction script.



## Running Model B

Model B enables the generation of novel materials from the learned latent space, and also supports pretraining on custom datasets.

### 1. Generate New Materials

    python modelB.py --mode generate --num_samples 1000

Notice that 
	
 •	You can change num_samples to generate more or fewer samples.
 
 •	Avoid setting num_samples too high (e.g., over 5000), as runtime depends on your system’s CPU/GPU.
 
 •	You may also configure other parameters. For example:

    python modelB.py --mode generate \
     --generate_data_csv path_to_your_csv.csv \
     --num_samples 1000 \
     --cond_melting XXX (use log value) \
     --cond_density XXX \
     --cond_form_energy XXX \
     --t XXX

### 2. Pretrain on Custom Data

To pretrain the model using your own dataset, use the following command:

    python modelB.py --mode pretrain \
     --pretrain_data_csv path_to_your_csv.csv \
     --epochs XXX \
     --batch_size XXX \
     --save_endecoder path_to_save_model.pt \
     --load_endecoder path_to_load_model.pt

Note: --load_endecoder and --save_endecoder can point to the same .pt file if you’re continuing training.
