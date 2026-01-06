# Deep Learning-Based Prediction of Protein Stability Changes (ΔΔG) for Insertion and Deletion Mutations

Master's thesis project implementing a deep learning framework to predict thermodynamic stability changes caused by insertion and deletion (indel) mutations in proteins.

## What this does

Predicts how insertion and deletion mutations affect protein stability (measured as ΔΔG in kcal/mol). Uses a combination of structural features, sequence properties, and protein language model embeddings. Separate neural network models for insertions vs deletions since they have different destabilization mechanisms.

## Results overview

**Training set (Tsuboyama et al. 2020) - 5-fold cross-validation:**
- Insertions: Pearson R = 0.653, RMSE = 1.45 kcal/mol
- Deletions: Pearson R = 0.534, RMSE = 1.23 kcal/mol

**Independent test (ProteinGym benchmark):**
- Insertions: R² = -1.77
- Deletions: R² = -1.30

The negative R² on ProteinGym indicates the model doesn't generalize well to new protein families. This appears related to limited training data diversity (diversity scores 0.009-0.017, below recommended thresholds) and domain shift between training and benchmark datasets.

## Repository structure

```
Model.ipynb                   Main pipeline (all code in one notebook)

Data folders:
PDBs/                        Training set protein structures
PDBs_ddG_benchmark/          Test structures for ΔΔG predictions
PDB_fitness_benchmark/       Test structures for fitness predictions

rSASA_results/               Relative solvent accessibility (training)
rSASA_results_ddG_benchmark/
rSASA_results_fitness_benchmark/

secondary_structure/         STRIDE secondary structure assignments (training)
secondary_structure_ddG_benchmark/
secondary_structure_fitness_benchmark/

dataset_benchmark/           Processed benchmark datasets
Useful_dataframe/           Training data and intermediate files
Indeli-E_benchmark_results/ External model comparison results

trained_models/             Saved model checkpoints
```

## Dependencies

Python 3.8+

Main packages:
```
tensorflow >= 2.4.0
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
biopython
tqdm
```

External tools needed:
- PyMOL (for rSASA calculations)
- STRIDE (secondary structure assignment): https://webclu.bio.wzw.tum.de/cgi-bin/stride/stridecgi.py

## Running the code

Everything is in `Model.ipynb`. The notebook is organized sequentially:

**Cells 0-3:** Data loading and initial feature extraction
- Loads training data from Tsuboyama dataset
- Processes IndelLM predictions
- Extracts physicochemical properties

**Cells 4-5:** Feature importance analysis
- Identifies top non-IndelLM features using Random Forest
- Key features: charged content, stability risk score, aromatic/proline content

**Cells 6-9:** Benchmark data preparation
- Processes ProteinGym benchmark datasets
- Extracts features for independent validation
- Handles both ΔΔG and fitness predictions

**Cells 10-13:** Data quality checks
- Protein-level independence verification
- Sequence similarity analysis between train/test
- Checks for potential data leakage

**Cells 14-15:** Model architecture and training functions
- Neural network with structural feature processing
- Attention mechanism implementation
- Ensemble training with 5-fold cross-validation

**Cell 19:** Main training execution
- Trains models for insertions and deletions
- Uses protein-level k-fold split
- Saves trained models to `trained_models/`

**Cells 20-21:** ΔΔG benchmark evaluation
- Tests on independent ProteinGym data
- Domain adaptation attempted
- Results show poor generalization

**Cells 22-24:** Fitness benchmark evaluation
- Alternative prediction target (fitness instead of ΔΔG)
- Similar generalization issues

**Cells 25-27:** Feature ablation analysis
- Tests contribution of feature groups
- Confirms language model embeddings are most important

**Cells 28-40:** Statistical analysis and comparisons
- Comparison with IndelLM and INDELi-E baselines
- Error distribution analysis
- Publication-quality figures

To run:
```bash
jupyter notebook Model.ipynb
# Or convert to script: jupyter nbconvert --to script Model.ipynb
```

Note: File paths are hardcoded to Windows paths (D:/Slides/master's_thesis/...). You'll need to update these in cells 19, 21, 24 to match your directory structure.

## Feature engineering

The model uses three types of features:

**Structural features** (~31% importance):
- Position-specific rSASA (relative solvent accessible surface area)
- Window-averaged rSASA (7-residue window)
- Secondary structure at mutation site (helix/sheet/coil encoded as 1.0/0.3/0.0)
- Window-averaged secondary structure

**Sequence properties** (~27% importance):
- Charged residue content
- Aromatic residue content  
- Proline content
- Polar residue content
- Combined pathogenicity scores

**Language model embeddings** (~42% importance):
- IndelLM Siamese embeddings
- Wild-type vs mutant sequence comparison
- 768-dimensional embedding vectors

Feature importance determined through Random Forest analysis and ablation testing.

## Model architecture

Implemented in TensorFlow/Keras. Key components:

```python
# Simplified version - see cell 15 for full implementation

class StructuralDataProcessor:
    # Extracts rSASA and secondary structure features
    # Handles window averaging and normalization

def build_attention_model(input_dim):
    # Fully connected layers + dropout
    # Multi-head attention mechanism
    # Separate architectures for insertions vs deletions
    
def train_with_protein_cv(data, n_splits=5):
    # Protein-level k-fold cross-validation
    # Prevents data leakage from homologous proteins
```

Training details:
- Optimizer: Adam
- Loss: MSE
- Batch size: 32
- Epochs: 120 (with early stopping)
- Regularization: Dropout (0.3), L2 weight decay

## Data sources

**Training:**
- Tsuboyama et al. (2020) - Deep mutational scanning of indel effects
- Original paper: PLoS Biol 18(3): e3000632
- ~1000 insertion mutations, ~800 deletion mutations

**Benchmarking:**
- ProteinGym (Notin et al. 2023) - Large-scale DMS benchmarks
- Downloaded from: https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/tree/main/indels
- Multiple protein families for independent validation

**Structures:**
- RCSB Protein Data Bank
- Retrieved using PDB IDs from experimental datasets

**Language model:**
- Modified IndelLM accessible via: [https://colab.research.google.com/drive/1mp4fc9P6Z0hMjC4jFPLi01R4Rvo00wEB](https://colab.research.google.com/drive/1mp4fc9P6Z0hMjC4iFPLi01R4Rvo00wEB#scrollTo=oTL-RmsdX9AR)

## Known issues and limitations

1. Poor generalization to new protein families - negative R² on independent test set indicates model is essentially failing on out-of-distribution data

2. Training data diversity is insufficient - calculated diversity scores (0.009 for insertions, 0.017 for deletions) fall well below recommended thresholds of 0.1-0.2

3. Domain shift problem - mean sequence identity between train and test is only 0.062-0.063, which is good for avoiding leakage but makes generalization very challenging

4. Limited structural coverage - training data dominated by a few protein folds, doesn't cover full structural diversity

5. Computational cost - embedding generation is slow, requires GPU for reasonable performance

6. Feature transferability - features that work for one protein family don't necessarily transfer to others

## What we learned

Despite poor benchmark performance, this work highlighted several important insights:

- Indel stability prediction is fundamentally harder than point mutation prediction
- Current training datasets are too limited for learning generalizable patterns
- Language models capture important patterns (42% feature importance) but aren't sufficient alone
- Proper cross-validation with protein-level splits is critical - sequence-level splits would give misleadingly optimistic results
- Need much more diverse experimental data spanning different protein folds and families

The negative results are themselves informative - they show that this problem requires either:
1. Significantly larger and more diverse training data, or
2. Different modeling approaches (e.g., physics-informed models, transfer learning from related tasks)

## References

Key papers:
1. Tsuboyama et al. (2020). A widespread family of heat-resistant obscure (Hero) proteins protect against protein instability and aggregation. PLoS Biol 18(3): e3000632
2. Notin et al. (2023). ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction. bioRxiv
3. Topolska et al. (2025). Deep indel mutagenesis reveals the impact of amino acid insertions and deletions on protein stability and function. Nat Commun 16(1): 2617

See thesis manuscript for complete bibliography.

## Contact

GitHub: @Haisu520

## License

Academic use only. Code provided as-is for research purposes.

---

Note: This is a Master's thesis project. Models should not be used for clinical or production applications without extensive additional validation.
