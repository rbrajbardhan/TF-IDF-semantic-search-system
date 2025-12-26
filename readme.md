# TF-IDF Semantic Search

This project implements a TF-IDF (Term Frequency-Inverse Document Frequency) based semantic search system for Rajya Sabha parliamentary questions and answers. It allows for efficient text retrieval and similarity matching using vectorized representations of documents.

## Features

- Preprocess and merge datasets
- Build and save TF-IDF models
- Evaluate retrieval performance
- Generate charts for analysis


## Installation

1. Clone the repository:

   ```
   git clone https://github.com/rbrajbardhan/TF-IDF-semantic-search-system
   cd TF-IDF_Semantic_Search
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The project uses the Rajya Sabha dataset from Kaggle: [Rajya Sabha Dataset](https://www.kaggle.com/datasets/rajanand/rajyasabha)

To set up the dataset:

1. Download the dataset files from the Kaggle link above.
2. Place the downloaded data files (e.g., CSV files) in the project root directory or as expected by the preprocessing scripts.

## Usage

### Building the Model

Run the model building script:

```
python model/build_and_save_model.py
```

### Running the Application

Start the main application:

```
python app.py
```

### Evaluation

Evaluate the retrieval system:

```
python evaluation/evaluate_retrieval.py
```

Generate charts:

```
python evaluation/generate_charts.py
```

### Preprocessing

Open and run the Jupyter notebook for data preprocessing:

```
jupyter notebook preprocessing/merge_datasets.ipynb
```

## Project Structure

- `app.py`: Main application entry point
- `requirements.txt`: Python dependencies
- `evaluation/`: Scripts for evaluating retrieval performance
  - `evaluate_retrieval.py`: Evaluation logic
  - `generate_charts.py`: Chart generation for results
- `model/`: Model building and saving
  - `build_and_save_model.py`: Script to build and save the TF-IDF model
- `preprocessing/`: Data preprocessing
  - `merge_datasets.ipynb`: Jupyter notebook for merging datasets

## Requirements

- Python 3.7+
- Libraries listed in `requirements.txt` (e.g., scikit-learn, pandas, numpy, matplotlib)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
