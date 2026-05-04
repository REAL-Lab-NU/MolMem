#!/bin/bash
# run_molecule_opt.sh - Run molecular optimization task

echo "Starting molecular optimization model training..."
python train.py --config-name molecule_opt

echo "Molecular optimization task completed!"
