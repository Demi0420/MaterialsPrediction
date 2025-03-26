# MaterialsPrediction

To get started:

1.	**Download** all the files to your local machine, and open [this Google Drive link](https://drive.google.com/file/d/1iBU7PA1sMc4bHE1RyUUUm-0JBAhmv-Wh/view?usp=share_link) to download data_all.csv file.
 
2.	Open a **terminal** in the project directory and run:

    ```bash
    pip install -r requirements.txt

This will install all the required libraries.




## Running the model

You can run the following commands to perform predictions for different material properties:
### 1.	Predict melting point:

    python3 modelA.py --target_col melting_point_log --abrev mp


### 2.	Predict density:

    python3 modelA.py --target_col density --abrev rho


### 3.	Predict formation energy per atom:

    python3 modelA.py --target_col formation_energy_per_atom --abrev fe



💡 Make sure your data files and model configurations are properly set up before running the script.
