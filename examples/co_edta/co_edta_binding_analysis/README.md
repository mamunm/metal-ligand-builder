# Metal-Ligand Binding Analysis Results
# Generated: 2025-06-03 11:51:20
# System: Co(edta)

## Directory Structure

```
co_edta_binding_analysis/
├── 01_initial_structures/
│   ├── metals/              # Initial metal geometries
│   ├── ligands/             # Ligand conformers
│   └── complexes/           # Metal-ligand poses
├── 02_optimized_structures/
│   ├── metals/              # XTB optimized metals
│   ├── ligands/             # XTB optimized ligands
│   └── complexes/           # XTB optimized complexes
├── 03_orca_inputs/
│   ├── metals/mult_*/       # ORCA inputs by multiplicity
│   ├── ligands/mult_*/
│   └── complexes/mult_*/
├── 04_reports/
│   ├── analysis_report.html # Comprehensive HTML report
│   ├── *_results.csv       # CSV summaries
│   └── *_top10.csv         # Best 10 structures
├── 05_best_structures/
│   ├── metals/              # 5 best metal structures
│   ├── ligands/             # 5 best ligand conformers
│   └── complexes/           # 5 best complexes
└── 06_metadata_files/
    ├── results_archive.json   # Complete results archive
    ├── workflow_summary.json  # Workflow summary
    └── *_metadata.json        # Various metadata files
```

## Key Files

- `06_metadata_files/results_archive.json`: Complete results in JSON format
- `06_metadata_files/workflow_summary.json`: High-level workflow summary
- `06_metadata_files/*_metadata.json`: Various metadata files
- `04_reports/analysis_report.html`: Comprehensive analysis report
- `05_best_structures/`: Easy access to best structures

## Analysis Summary

- Metals: 1/1 successful optimizations
- Ligands: 5/5 successful optimizations
- Complexes: 46/50 successful optimizations
- ORCA inputs generated: 6 files