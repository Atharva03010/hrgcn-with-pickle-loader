# HRGCN (Modified)

This repository is a modified version of the [original HRGCN repository](https://github.com/jiaxililearn/HRGCN). It extends the original functionality by adding the ability to load data directly from pickle files.

> **Note:** This code is configured to load a specific graph dataset structure and is **not** a general-purpose implementation of HRGCN.

## Features
* **Pickle Support:** Load graph data via custom pickle files.
* **Anomaly Detection:** Calculates anomaly scores for individual nodes.
* **SVDD Integration:** Computes distances from the Support Vector Data Description (SVDD) center. Exclusively used unsupervised approach instead of the hybrid semi-supervised approach of the original implementation.

## Usage

To run the script, use the following command:

```bash
python src/main.py --pickle_path /path/to/picklefile
```

## Results
Script writes anomaly scores of indiviual nodes to a csv file. The final average distance from SVDD center is printed after testing.

## Requirements
Same as the original implementation

