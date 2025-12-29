# Deep Learning-Based Prediction of Protein Stability Changes (ΔΔG) for Insertion and Deletion Mutations

## Overview

This repository contains the complete implementation of a deep learning framework for predicting thermodynamic stability changes (ΔΔG) caused by insertion and deletion (indel) mutations in proteins. This work was developed as part of a Master's thesis project.

### Key Features

- Specialized neural network architectures for insertions vs. deletions
- Integration of multiple feature types:
  - Structural features (rSASA, secondary structure)
  - Protein language model embeddings (IndelLM)
  - Evolutionary and physicochemical properties
- Rigorous protein-level cross-validation
- Comprehensive benchmarking on ProteinGym dataset

### Main Results

**Training Performance (Tsuboyama dataset):**
- Insertions: Pearson R = 0.653, RMSE = 1.45 kcal/mol
- Deletions: Pearson R = 0.534, RMSE = 1.23 kcal/mol

**Independent Validation (ProteinGym):**
- Revealed significant generalization challenges (R² < 0), highlighting the difficulty of cross-family prediction for indel mutations

## Repository Structure

```
├── Model.ipynb                          # Main pipeline: feature extraction, training, evaluation
├── README.md                            # This file
│
├── PDBs/                                # Protein structure files (training set)
├── PDBs_ddG_benchmark/                  # PDB files for ΔΔG benchmark
├── PDB_fitness_benchmark/               # PDB files for fitness benchmark
│
├── secondary_structure/                 # STRIDE secondary structure assignments (training)
├── secondary_structure_ddG_benchmark/   # Secondary structures (ΔΔG benchmark)
├── secondary_structure_fitness_benchmark/ # Secondary structures (fitness benchmark)
│
├── rSASA_results/                       # Relative SASA values (training)
├── rSASA_results_ddG_benchmark/         # rSASA (ΔΔG benchmark)
├── rSASA_results_fitness_benchmark/     # rSASA (fitness benchmark)
│
├── dataset_benchmark/                   # Processed benchmark datasets
├── Useful_dataframe/                    # Intermediate data files
├── Indeli-E_benchmark_results/          # Evaluation results
│
└── trained_models/                      # Saved model checkpoints
```

## Requirements

### Software Dependencies

```bash
python >= 3.8
torch >= 1.10.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
biopython >= 1.79
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### External Tools

- **PyMOL** (for rSASA calculation)
- **STRIDE** (for secondary structure assignment)
  - Web server: https://webclu.bio.wzw.tum.de/cgi-bin/stride/stridecgi.py
  - Or local installation

### Pre-trained Models

The modified IndelLM model used for embeddings can be accessed via:
- [Google Colab Link](https://colab.research.google.com/drive/1mp4fc9P6Z0hMjC4jFPLi01R4Rvo00wEB#scrollTo=4XuDbBG-GBFd)

## Installation

```bash
# Clone repository
git clone https://github.com/Haisu520/Deep-Learning-Based-Prediction-of-Protein-Stability-Changes-G-for-Insertion-and-Deletion-Mutations.git
cd Deep-Learning-Based-Prediction-of-Protein-Stability-Changes-G-for-Insertion-and-Deletion-Mutations

# Install dependencies
pip install -r requirements.txt  # Create this file with above dependencies

# Download pre-trained IndelLM (if needed)
# Follow instructions in Google Colab link
```

## Usage

### Quick Start

The main pipeline is contained in `Model.ipynb`. To reproduce the analysis:

1. **Data Preparation** (Sections 1-3 in notebook)
   - Load training data from Tsuboyama et al.
   - Extract structural features (rSASA, secondary structure)
   - Generate IndelLM embeddings

2. **Model Training** (Sections 4-5)
   - Separate models for insertions and deletions
   - Protein-level k-fold cross-validation
   - Hyperparameter optimization

3. **Evaluation** (Sections 6-7)
   - Cross-validation performance metrics
   - Independent validation on ProteinGym
   - Feature importance analysis

### Running the Pipeline

```bash
# Open Jupyter notebook
jupyter notebook Model.ipynb

# Or run as script (if converted)
python model_script.py
```

### Computing Structural Features

If you need to recompute structural features:

**rSASA Calculation:**
```python
# Using PyMOL (script included in notebook)
pymol -c -r Get_rSASA_auto.py -- <pdb_file>
```

**Secondary Structure:**
```bash
# Submit PDB to STRIDE web server or use local installation
stride <pdb_file> > secondary_structure.txt
```

## Data Sources

### Training Data
- **Tsuboyama et al. (2020)** - Deep mutational scanning of indels
- Available in: `Useful_dataframe/`

### Benchmark Data
- **ProteinGym** (Notin et al., 2023) - Large-scale DMS benchmarks
  - Downloaded from: https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/tree/main/indels
  - Processed files in: `dataset_benchmark/`

### Structural Data
- **RCSB PDB** - 3D protein structures
- Retrieved using PDB IDs from training/benchmark datasets

## Key Implementation Details

### Model Architecture
- **Base**: Fully connected neural networks with dropout
- **Attention**: Multi-head attention over sequence positions
- **Separate models**: One for insertions, one for deletions
- **Ensemble**: Multiple models combined for final prediction

### Feature Engineering
1. **Structural Features** (31% importance)
   - rSASA (relative solvent accessible surface area)
   - Secondary structure (helix, sheet, coil)
   
2. **Language Model Embeddings** (42% importance)
   - IndelLM Siamese embeddings
   - Wild-type and mutant representations
   
3. **Sequence Context** (27% importance)
   - Proline content, aromatic content
   - Charged residue content
   - Combined pathogenicity scores

### Cross-Validation Strategy
- **Protein-level splits**: Ensures no protein appears in both train and test
- **5-fold CV**: For robust performance estimation
- **Prevents data leakage**: Critical for evolutionary-related proteins

## Results and Analysis

### Performance Summary

| Mutation Type | Dataset | Metric | Value |
|--------------|---------|--------|-------|
| Insertions | Tsuboyama (CV) | Pearson R | 0.653 |
| Insertions | Tsuboyama (CV) | RMSE | 1.45 kcal/mol |
| Deletions | Tsuboyama (CV) | Pearson R | 0.534 |
| Deletions | Tsuboyama (CV) | RMSE | 1.23 kcal/mol |
| Insertions | ProteinGym | R² | -1.77 |
| Deletions | ProteinGym | R² | -1.30 |

### Key Findings

✅ **Successful aspects:**
- Proof-of-concept for specialized indel stability prediction
- Integration of multiple complementary features
- Proper validation methodology

⚠️ **Limitations identified:**
- Poor generalization to unseen protein families
- Training data diversity insufficient (diversity scores: 0.009-0.017)
- Domain shift between training and benchmark datasets

### Lessons Learned

The negative R² on independent validation highlights:
1. Need for more diverse training data across protein folds
2. Challenges in feature transferability across protein families
3. Importance of domain-aware validation strategies

## Citation

If you use this code or approach in your research, please cite:

```bibtex
@mastersthesis{yourname2025indel,
  title={Deep Learning-Based Prediction of Protein Stability Changes for Insertion and Deletion Mutations},
  author={Your Name},
  school={Your University},
  year={2025}
}
```

### Key References

1. Tsuboyama, K., et al. (2020). A widespread family of heat-resistant obscure (Hero) proteins. *PLoS Biol*, 18(3).
2. Notin, P., et al. (2023). ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction. *bioRxiv*.
3. Topolska, M., et al. (2025). Deep indel mutagenesis reveals the impact of insertions and deletions. *Nat Commun*, 16(1).

## Known Issues and Future Work

### Current Limitations
- Model overfits to training protein families
- Limited structural coverage in training data
- Computational cost of embedding generation

### Planned Improvements
- Expand training data with diverse protein folds
- Implement domain adaptation techniques
- Explore graph neural networks for structural encoding
- Add uncertainty quantification

## Contact

- **GitHub**: [@Haisu520](https://github.com/Haisu520)
- **Email**: [Your email if you want to include]

## License

[Specify license - typically MIT or GPL for academic code]

## Acknowledgments

- IndelLM authors for the pre-trained language model
- ProteinGym team for benchmark datasets
- [Your supervisor/institution]

---

**Note**: This is an academic research project. The models are provided for research purposes and should not be used for clinical applications without extensive additional validation.
