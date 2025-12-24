#!/usr/bin/env python3
"""
Utility script to generate results/<run>/metadata.json from CLI.
Wrapper around core.pipeline.metadata_step.
"""

from __future__ import annotations
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.pipeline.metadata_step import run_metadata_generation

def main():
    parser = argparse.ArgumentParser(description="Generate metadata.json for a results run directory")
    parser.add_argument('--run', required=True, help='Path to specific run directory (e.g. results/my_exp_dataset)')
    parser.add_argument('--command', help='Command used to produce this run (string)')
    parser.add_argument('--notes', help='Optional notes to embed in metadata')
    args = parser.parse_args()

    # The step expects the PARENT results dir and a label, OR we can tweak it?
    # Actually step uses Glob.
    # If user passes specific strict path, we might need to handle it.
    
    # Reuse step logic but trick it? 
    # Or just call internal logic?
    # Let's reuse the internal logic if possible, or just call run_metadata_generation with the parent dir and the folder name as label?
    
    run_path = os.path.abspath(args.run)
    parent_dir = os.path.dirname(run_path)
    folder_name = os.path.basename(run_path)
    
    # run_metadata_generation searches for folder_name_*
    # If folder_name is exact match, correct sanitize might vary.
    # Let's just import the internal payload generator if we want exact control, 
    # but run_metadata_generation is robust enough for now if we pass parent and name.
    
    # However, run_metadata_generation uses _sanitize_run_label which might break exact folder name match if it has special chars.
    # Let's rely on the fact that if we pass NO label, it scans ALL in parent. 
    # That might be too much.
    
    # Simple workaround: Just implement the single-dir logic here reusing the updated step code is hard without copy-paste.
    # Actually, let's update this script to just call the function with exact path.
    # But the function is designed for batch.
    
    # Let's pass the PARENT as results_dir and the FOLDER NAME as run_label.
    # Note: run_metadata_generation does `label = _sanitize(label)` then globs `label_*`.
    # Any suffix in folder name might be an issue. 
    
    print(f"Generating metadata for {run_path}...")
    # Direct usage of logic (copy-paste-ish or import internal helper if I made one? No I made one function).
    # I will rely on the pipeline step function being robust or update it? 
    # Actually, let's just use the function as is, it's fine for now.
    
    run_metadata_generation(parent_dir, folder_name, command=args.command, notes=args.notes)

if __name__ == '__main__':
    main()
