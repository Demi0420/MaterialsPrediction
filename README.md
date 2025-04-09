# MaterialsPrediction

This repository contains code for training and generating material property predictions using machine learning models.

## Getting Started

1. **Download** all the files in this repository to your local machine.

2. Download the dataset file `data_all.csv` from the following [Google Drive link](https://drive.google.com/file/d/1iBU7PA1sMc4bHE1RyUUUm-0JBAhmv-Wh/view?usp=share_link).

3. Open a **terminal** in the project directory and run:

   ```bash
   pip install -r requirements.txt

This command will install all required Python libraries.

⚠️ If you need to use the CUDA-enabled version of PyTorch, please install it manually according to your system’s GPU configuration.

4.	Before running the code, place your data_all.csv file in the `data_csv/` folder.
Alternatively, you can provide a custom path using the corresponding arguments:

	•	For Model A, modify the path directly in the code.

	•	For Model B, use `--pretrain_data_csv` or `--generate_data_csv`.



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

Let me know if you'd like to add a `License`, `Citation`, `Examples`, or `Contributing` section too!
