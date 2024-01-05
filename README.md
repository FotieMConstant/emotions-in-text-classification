# Emotions in Text Classification with BERT

This project aims to classify emotions in text using a DistilBERT-based model. It includes both a Jupyter notebook for training and a Django API for serving predictions.

## Getting Started

Follow the steps below to set up and run the project.

### Prerequisites

- Python 3.10.3 (Anaconda or Miniconda recommended)
- [Jupyter Notebook](https://jupyter.org/install)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/FotieMConstant/emotions-in-text-classification.git
    cd emotions-in-text-classification
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    conda create --name myenv python=3.10.3
    conda activate myenv
    ```

3. Install project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

1. Run the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

2. Open and run the `train_model.ipynb` notebook. This will generate the production model (`dbert_model.h5`).

### Running the Django API

1. Navigate to the Django API directory:

    ```bash
    cd emotion_api
    ```

2. Run the Django development server:

    ```bash
    python manage.py runserver
    ```

3. The API will be accessible at `http://localhost:8000/api/predict/`.

## Usage

You can make predictions by sending a POST request to the API endpoint with the text you want to classify.

Example using [curl](https://curl.se/):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Your text here"}' http://localhost:8000/api/predict/
```
* You can equally just use [postman](https://postman.com/) to make the request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE)  file for details.