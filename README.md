# NLP Assignment 3: Fact-Checking with Word Overlap Baseline

This repository contains the implementation for **NLP Assignment 3**, focusing on fact-checking using a word overlap baseline model. The goal is to classify factual claims as `supported` or `not supported` by analyzing overlap scores between claims and retrieved evidence passages.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
Fact-checking is a crucial task in NLP, particularly for detecting misinformation. This project implements a word overlap baseline model to classify factual claims. It uses a Bag-of-Words representation to compute similarity scores between claims and evidence passages. The model predicts `supported` or `not supported` based on these scores, providing a simple yet effective baseline for comparison.

## Features
- **Word Overlap Scoring:** Computes overlap scores between claims and evidence passages using the Bag-of-Words approach.
- **Threshold-Based Classification:** Classifies claims based on a similarity score threshold.
- **Evaluation Metrics:** Reports accuracy, precision, recall, and F1-score for performance evaluation.
- **Human-Readable Output:** Displays predictions for each claim along with corresponding scores.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/NLP-A3.git
cd NLP-A3
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the main script as follows:

```bash
python fact_checker.py --claims_file <path_to_claims_data> --evidence_file <path_to_evidence_data> --threshold <similarity_threshold>
```

#### Command-line Arguments
- `--claims_file`: Path to the file containing factual claims.
- `--evidence_file`: Path to the file containing evidence passages.
- `--threshold`: Similarity score threshold for classifying claims as `supported` (default: 0.5).

Example:
```bash
python fact_checker.py --claims_file data/claims.csv --evidence_file data/evidence.csv --threshold 0.6
```

## Example Output
The script will output predictions for each claim along with evaluation metrics for the overall performance:

Sample output:
```
Claim 1: Supported (Score: 0.72)
Claim 2: Not Supported (Score: 0.45)
...
Accuracy: 74.2%
Precision for 'Supported': 80.9%
Recall for 'Supported': 64.3%
F1-Score for 'Supported': 71.6%
```

## Dependencies
This implementation uses the following dependencies:
- `numpy`
- `nltk`

Install the dependencies using:
```bash
pip install numpy nltk
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to use, extend, and contribute to this project. Improvements and suggestions are always welcome!
