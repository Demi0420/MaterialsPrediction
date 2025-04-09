# MaterialsPrediction

To get started:

1.	**Download** all the files to your local machine, and open [this Google Drive link](https://drive.google.com/file/d/1iBU7PA1sMc4bHE1RyUUUm-0JBAhmv-Wh/view?usp=share_link) to download data_all.csv file.
 
2.	Open a **terminal** in the project directory and run:

    ```bash
    pip install -r requirements.txt

This will install all the required libraries. If you need to install the torch-cuda version, please manually install.




## Running the model A

You can run the following commands to perform predictions for different material properties:
### 1.	Predict melting point:

    python modelA.py --target_col melting_point_log --abrev mp


### 2.	Predict density:

    python modelA.py --target_col density --abrev rho


### 3.	Predict formation energy per atom:

    python modelA.py --target_col formation_energy_per_atom --abrev fe



💡 Make sure your data files and model configurations are properly set up before running the script.


## Running the model B
You can run the following command to generate new materials from the latent space:
### Generate:

    python modelB.py --mode generate --num_samples 1000

You can change the value of num_samples, but please notice do not make it too large (like over 5000), because it depends on your computer's CPU or GPU version, it may take sometime to give the results.
Also, you can change other parameters, please refer to the py file. For example,
python modelB.py --mode generate --generate_data_csv path_to_your_csv(end with .csv) --num_samples 1000 --cond_melting XXX(please use log number) --cond_density XXX --cond_form_energy XXX --t XXX


If you want to pretrain the model with other data files, please run the following command
### Pretrain:

    python modelB.py --mode pretrain --pretrain_data_csv path_to_your_csv(end with .csv) --epochs XXX --batch_size XXX --save_endecoder path_to_your_endecoder_weights(end with .pt) --load_endecoder path_to_your_endecoder_weights(end with .pt, the same as --save_endecoder)

