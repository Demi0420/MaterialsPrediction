# MaterialsPrediction

To get started:
	1.	<strong>Download</strong> all the files to your local machine.
	2.	Open a <strong>terminal</strong> in the project directory and run:

<code>pip install -r requirements.txt</code>

This will install all the required libraries.

## Running the model

You can run the following commands to perform predictions for different material properties:
	### 1.	Predict melting point:

<code>python3 modelA.py --target_col melting_point_log --abrev mp</code>


	### 2.	Predict density:

<code>python3 modelA.py --target_col density --abrev rho</code>


	### 3.	Predict formation energy per atom:

<code>python3 modelA.py --target_col formation_energy_per_atom --abrev fe</code>



	💡 Make sure your data files and model configurations are properly set up before running the script.
