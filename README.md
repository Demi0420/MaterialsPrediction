# MaterialsPrediction

This repository contains the code, trained model weights, data files, generated structures, and additional experimental results associated with the revised manuscript.

The project mainly includes two models:

- `modelA.py` (SC-CGGN): a latent-space-based generative model for proposing new candidate material structures. It also supports model pretraining.
- `modelB.py` (CGPR): a CGCNN-based property prediction model for predicting material properties such as melting point, density, and formation energy.

In the revised version, we additionally provide model variants, newly trained model weights, benchmark scripts, prediction scripts, and the 20 shortlisted pre-DFT CIF files corresponding to Table 2 in the revised manuscript.


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
```

## 2. Requirements

Install the required dependencies using:

	```bash
	pip install -r requirements.txt

If GPU acceleration is used, please make sure that PyTorch is installed with the CUDA version compatible with the local environment.

CUDA availability can be checked by running:

	```bash
	python test_cuda.py


## 3. Data files

The main data files are stored in the data_csv/ directory:

```text
data_csv/
├── data_all.csv
├── data_all_with_volume.csv
└── data_e43V.csv
```
Download links:

1. `data_all.csv` (Full dataset used for ModelB pretraining) [Google Drive link](https://drive.google.com/file/d/1y8gZ2XQ4yVoF49rN4mG1BIXX0_f1PXN2/view?usp=share_link).

2. `data_e43V.csv` (Filtered material dataset used for fine-tuning, material generation, and benchmark analyses) [Google Drive link](https://drive.google.com/file/d/1y8gZ2XQ4yVoF49rN4mG1BIXX0_f1PXN2/view?usp=share_link).


## 4. Model A: SC-CGGN (latent-space-based generation model)

`modelA.py` implements the original latent-space-based generative model, SC-CGGN, used to propose new candidate material structures. The model supports both pretraining and conditional guided generation from the learned latent space.

The main trained weight file for the original Model A is:

```text
modelA-weights/endecoder_model.pt
```

This weight file is used in the original guided generation workflow.

### 4.1 Generate new candidate materials

To generate new candidate materials using the trained Model A, run:

```bash
python modelA.py \
  --mode generate \
  --num_samples 1000
```

Notes:

- `--num_samples` controls the number of generated samples.
- Avoid setting `--num_samples` too high, for example above 5000, because the runtime depends on the available CPU/GPU resources.
- If not explicitly specified, the script uses the default generation settings defined in `modelA.py`.

Other generation parameters can also be configured manually. For example:

```bash
python modelA.py \
  --mode generate \
  --generate_data_csv data_csv/data_e43V.csv \
  --num_samples 1000 \
  --cond_melting XXX \
  --cond_density XXX \
  --cond_form_energy XXX \
  --t XXX
```

Notes:

- `--generate_data_csv` specifies the dataset used during generation.
- `--cond_melting` specifies the melting-point condition. This value should be given on the logarithmic scale.
- `--cond_density` specifies the density condition.
- `--cond_form_energy` specifies the formation-energy condition.
- `--t` controls the sampling or generation temperature parameter used in the generation process.

For example, if the target melting point is 1400 K, the corresponding logarithmic value is approximately:

```text
log(1400) = 7.244227516
```

### 4.2 Pretrain Model A on a custom dataset

To pretrain Model A using a custom dataset, run:

```bash
python modelA.py \
  --mode pretrain \
  --pretrain_data_csv data_csv/data_e43V.csv \
  --epochs 200 \
  --batch_size 32 \
  --save_endecoder path_to_save_model.pt \
  --load_endecoder path_to_load_model.pt
```

The arguments are:

| Argument | Description |
|---|---|
| `--mode pretrain` | Runs Model A in pretraining mode |
| `--pretrain_data_csv` | CSV file used for pretraining |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size used during training |
| `--save_endecoder` | Path for saving the trained encoder-decoder model |
| `--load_endecoder` | Path for loading an existing encoder-decoder model |

Note that `--load_endecoder` and `--save_endecoder` can point to the same `.pt` file if continuing training from an existing checkpoint.


## 5. Prototype-based variant of Model A

In the revised version, we additionally provide a prototype-based variant of Model A:

```text
model_prototype.py
modelA-weights/prototype_model.pth
```

The files are used as follows:

| File | Description |
|---|---|
| `model_prototype.py` | Prototype-based variant of Model A |
| `modelA-weights/prototype_model.pth` | Trained weights for the prototype-based Model A variant |

This variant was used to evaluate whether prototype-based latent-space generation can provide an alternative strategy for generating candidate materials.

### 5.1 Training the prototype-based Model A

```bash
python model_prototype.py \
  --mode train \
  --data_csv data_csv/data_e43V.csv \
  --epochs 200 \
  --batch_size 32 \
  --save_model modelA-weights/prototype_model.pth
```

### 5.2 Generating candidates using the prototype-based Model A

```bash
python model_prototype.py \
  --mode generate \
  --data_csv data_csv/data_e43V.csv \
  --load_model modelA-weights/prototype_model.pth \
  --num_samples 1000
```

The generated results are saved to the output file specified in the script.

## 6. Model B: CGPR (CGCNN-based property prediction model)

`modelB.py` implements the CGCNN-based property prediction model used to predict material properties, including:
- melting point
- density
- formation energy

In the revised version, we provide the following fine-tuned Model B weights:

```text
modelB-weights/best_modelB_mp_finetuned.pth
modelB-weights/best_modelB_rho_finetuned.pth
```

The files are used as follows:

| File | Description |
|---|---|
| `modelB-weights/best_modelB_mp_finetuned.pth` | Fine-tuned Model B weights for melting-point prediction |
| `modelB-weights/best_modelB_rho_finetuned.pth` | Fine-tuned Model B weights for density prediction |

These weights were obtained after pretraining followed by fine-tuning.

For the prediction of newly generated structures using `predictB.py`, the provided fine-tuned Model B weights are loaded with `use_extra_fea=False`. Therefore, the prediction input only requires structural information and does not require other extra feature columns.


## 7. Formula-grouped variant of Model B

In the revised version, we also provide a formula-grouped variant of Model B:

```text
modelB-formula-group.py
modelB-weights/best_modelB_mp_finetuned_grouped.pth

```

In this variant, the data grouping strategy was modified so that entries with the same chemical formula are grouped during data splitting. This prevents the same formula from appearing in both the training and test sets. Apart from this data grouping strategy, the model architecture and main training procedure were kept unchanged.

This variant was used to evaluate the influence of formula-based grouping on melting-point prediction performance.


### 7.1 Training the formula-grouped melting-point prediction model

```bash
python modelB-formula-group.py \
  --target_col melting_point_log \
  --abrev mp \
  --data_all_csv data_csv/data_all.csv \
  --data_allowed_csv data_csv/data_e43V.csv \
  --seed 42
```

## 8. Prediction for generated structures using Model B

`predictB.py` predicts the properties of newly generated materials that contain only structural information. The script loads the fine-tuned Model B weights and predicts:

- melting point
- density

The melting-point model predicts `melting_point_log`, and the prediction is transformed back to the original melting-point scale using an exponential transformation.

### 8.1 Input file format

The input CSV file should contain at least the following columns:

```text
formula_pretty
structure
```

The columns are defined as follows:

| Column | Description |
|---|---|
| `formula_pretty` | Chemical formula of the material |
| `structure` | Structure dictionary saved using `pymatgen.core.Structure.as_dict()` and stored as a JSON string |

Because the provided fine-tuned Model B weights are loaded with `use_extra_fea=False`, the input file does not need to contain `volume`, `volume_per_atom`, or any other extra feature columns.

### 8.2 Running prediction

```bash
python predictB.py \
  --input_csv path_to_your_csv \
  --output_csv path_to_your_csv \
  --mp_model modelB-weights/best_modelB_mp_finetuned.pth \
  --rho_model modelB-weights/best_modelB_rho_finetuned.pth
```

## 9. Benchmarking generation variants

`benchmark_variants.py` provides benchmark workflows for comparing different Model A generation variants. This script was used for additional experiments related to the revised manuscript and reviewer comments.

The script supports three generation variants:

```text
full_guided
concat_only
unconditional
```

It also supports two actions:

```text
generate
train_unconditional
```

### 9.1 Variants

#### full_guided

`full_guided` corresponds to the original conditional guided generation setting. This mode uses the original Model A guided generation function and generates candidate materials under specified melting-point and density conditions.

#### concat_only

`concat_only` is a benchmark-only variant. It keeps the decoder of the full model but decodes randomly sampled latent vectors without performing the full guided latent optimization. This variant was used to evaluate the contribution of the full guided generation strategy.

#### unconditional

`unconditional` is a benchmark-only unconditional generation variant. It removes the extra features or conditional information and generates structures only from the latent representation. This variant was used to evaluate the contribution of conditional information.



### 9.2 Full-guided generation

```bash
python benchmark_variants.py \
  --model_module_path modelA.py \
  --variant full_guided \
  --action generate \
  --weights modelA-weights/endecoder_model.pt \
  --data_csv data_csv/data_e43V.csv \
  --num_samples 1000 \
  --cond_melting 7.244227516 \
  --cond_density 8.0 \
  --output_dir benchmark_outputs/full_guided
```

Notes:
- `--cond_melting` is given on the logarithmic scale.
- `7.244227516` is approximately equal to `log(1400)`.
- `--cond_density 8.0` indicates a density condition of 8.0.


### 9.3 Concat-only generation

```bash
python benchmark_variants.py \
  --model_module_path modelA.py \
  --variant concat_only \
  --action generate \
  --weights modelA-weights/endecoder_model.pt \
  --data_csv data_csv/data_e43V.csv \
  --num_samples 1000 \
  --output_dir benchmark_outputs/concat_only
```

The output files usually include:

```text
benchmark_outputs/concat_only/generated_CIF/
benchmark_outputs/concat_only/generated_metadata.csv
```

This variant uses the same decoder weights as the full model but does not perform full guided latent optimization.


### 9.4 Training the unconditional benchmark model

The `unconditional` variant needs to be trained separately:

```bash
python benchmark_variants.py \
  --model_module_path modelA.py \
  --variant unconditional \
  --action train_unconditional \
  --data_csv data_csv/data_e43V.csv \
  --epochs 300 \
  --batch_size 32 \
  --weights benchmark_outputs/unconditional/unconditional_model.pt \
  --output_dir benchmark_outputs/unconditional
```

This command trains the benchmark-only unconditional generation model and saves the weights to:

```text
benchmark_outputs/unconditional/unconditional_model.pt
```

### 9.5 Generating structures using the unconditional benchmark model

After training the unconditional model, structures can be generated using:

```bash
python benchmark_variants.py \
  --model_module_path modelA.py \
  --variant unconditional \
  --action generate \
  --weights benchmark_outputs/unconditional/unconditional_model.pt \
  --data_csv data_csv/data_e43V.csv \
  --num_samples 1000 \
  --output_dir benchmark_outputs/unconditional
```

The output files usually include:

```text
benchmark_outputs/unconditional/generated_CIF/
benchmark_outputs/unconditional/generated_metadata.csv
```

## 10. Shortlisted pre-DFT CIF files for Table 2

In the revised version, we provide the 20 shortlisted pre-DFT CIF files corresponding to Table 2 in the revised manuscript.

These files are stored in:

```text
table2_shortlisted_pre_dft_cifs/
```

Example files include:

```text
1_SrZr7Ge6.cif
2_RbSr(ZrGe)6.cif
...
```

These CIF files represent the generated candidate structures before DFT relaxation or calculation. They were used as the input structures for the subsequent DFT validation workflow.


## 11. DFT-related workflow

For the DFT validation step, we provide the scripts used to prepare Quantum ESPRESSO input files from generated CIF structures and to generate additional CIF variants for the robustness analysis described in the response to Comment2.4.

We do not include all Quantum ESPRESSO `.in` and `.out` files in this repository. Instead, we provide the input CIF files and the scripts used to generate the corresponding Quantum ESPRESSO input files. This keeps the repository compact while preserving the reproducibility of the DFT input preparation workflow.

The DFT-related scripts are stored in:

```text
dft/
├── prepare_qe_inputs.py
└── make_comment4_variants.py
```

### 11.1 Preparing Quantum ESPRESSO input files from CIF files

`dft/prepare_qe_inputs.py` converts CIF files into Quantum ESPRESSO input files for variable-cell relaxation (`vc-relax`).

The script reads CIF files using `pymatgen` with:

```python
CifParser(str(cif_path)).get_structures(primitive=False)
```

Thus, the conventional structure is kept during CIF parsing.

To generate Quantum ESPRESSO `.in` files from CIF files, run:

```bash
python dft/prepare_qe_inputs.py \
  --input_cif_dir table2_shortlisted_pre_dft_cifs \
  --output_dir qe_inputs/table2_shortlisted \
  --pseudo_dir path_to_pseudopotentials \
  --qe_outdir qe_tmp
```

The arguments are:

| Argument | Description |
|---|---|
| `--input_cif_dir` | Directory containing input CIF files |
| `--output_dir` | Directory where the generated Quantum ESPRESSO `.in` files will be saved |
| `--pseudo_dir` | Directory containing the pseudopotential files used by Quantum ESPRESSO |
| `--qe_outdir` | Scratch/output directory used by Quantum ESPRESSO during calculation |


### 11.2 Main DFT input settings

The main DFT input settings used in `prepare_qe_inputs.py` are:

```text
calculation      = 'vc-relax'
verbosity        = 'high'
tstress          = .true.
tprnfor          = .true.

ecutwfc          = 40.0
ecutrho          = 320.0
occupations      = 'smearing'
degauss          = 0.02
smearing         = 'mp'
ibrav            = 0

conv_thr         = 0.0001
mixing_mode      = 'local-TF'
mixing_beta      = 0.3

ion_dynamics     = 'bfgs'
cell_dynamics    = 'bfgs'
cell_factor      = 2.0
press_conv_thr   = 0.5
K_POINTS         = 6 6 6 0 0 0
```

The generated input files include:

```text
&CONTROL
&SYSTEM
&ELECTRONS
&IONS
&CELL
ATOMIC_SPECIES
K_POINTS automatic
CELL_PARAMETERS angstrom
ATOMIC_POSITIONS angstrom
```

The pseudopotential file names are specified in the `PSEUDO_MAP` dictionary in `prepare_qe_inputs.py`.


These CIF files represent the generated candidate structures before DFT relaxation or calculation. They were used as input structures for the subsequent DFT validation workflow.

To prepare Quantum ESPRESSO input files for these structures, run:

```bash
python dft/prepare_qe_inputs.py \
  --input_cif_dir table2_shortlisted_pre_dft_cifs \
  --output_dir qe_inputs/table2_shortlisted \
  --pseudo_dir path_to_pseudopotentials \
  --qe_outdir qe_tmp
```

### 11.4 CIF variants for the robustness analysis in response to Comment #4

We also provide the script used to generate CIF variants for the additional robustness analysis described in the response to Comment #4:

```text
dft/make_comment4_variants.py
```

For each input CIF file, the script generates the following seven structures:

```text
orig.cif
coord_005_a.cif
coord_005_b.cif
coord_010_a.cif
coord_010_b.cif
lat_1pct_1deg.cif
lat_3pct_3deg.cif
```

These variants include:

| Variant | Description |
|---|---|
| `orig.cif` | Original input CIF structure |
| `coord_005_a.cif` | Structure with random Cartesian coordinate perturbation of 0.05 Å |
| `coord_005_b.cif` | Another structure with random Cartesian coordinate perturbation of 0.05 Å |
| `coord_010_a.cif` | Structure with random Cartesian coordinate perturbation of 0.10 Å |
| `coord_010_b.cif` | Another structure with random Cartesian coordinate perturbation of 0.10 Å |
| `lat_1pct_1deg.cif` | Structure with random lattice length perturbation of ±1% and angle perturbation of ±1° |
| `lat_3pct_3deg.cif` | Structure with random lattice length perturbation of ±3% and angle perturbation of ±3° |

To generate the CIF variants, run:

```bash
python dft/make_comment4_variants.py \
  --input_dir path_to_original_cifs \
  --output_dir comment4_cif_variants \
  --base_seed 20260330
```

The generated CIF variants can then be converted into Quantum ESPRESSO input files using `prepare_qe_inputs.py`:

```bash
python dft/prepare_qe_inputs.py \
  --input_cif_dir comment4_cif_variants \
  --output_dir qe_inputs/comment4_cif_variants \
  --pseudo_dir path_to_pseudopotentials \
  --qe_outdir qe_tmp
```

### 11.5 Notes on DFT input and output files

This repository does not include all generated Quantum ESPRESSO `.in` or `.out` files.

The `.in` files can be regenerated from the provided CIF files using:

```text
dft/prepare_qe_inputs.py
```

The `.out` files are not included because they are large and system-dependent. The provided scripts and CIF files are sufficient to reproduce the DFT input preparation workflow.
