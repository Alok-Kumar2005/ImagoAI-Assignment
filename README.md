# ImagoAI-Assignment

```
conda create -n ImageAI python=3.10 -y
```
```
conda activate ImageAI
```
```
pip install -r requirements.txt
```

# ImagoAI Assignment

This repository documents the end-to-end process for data preprocessing, dimensionality reduction, model training, and evaluation for our hyperspectral corn dataset.

## Data Preprocessing

1. **Repository Setup and Data Loading:**  
   - A repository was created, and the dataset was loaded into it.
   - The data is mostly preprocessed; however, some outliers are still present.

2. **Outlier Removal:**  
   - Outliers were removed using the IQR (Interquartile Range) method.
   - This step helped to mitigate the influence of extreme values on subsequent analyses.

3. **Normalization:**  
   - The dataset is largely normalized, although some rows are not fully normalized.
   - Detailed visualizations and analysis of the preprocessing steps can be found in the `research/trails.ipynb` notebook.

## Dimensionality Reduction

Given that the dataset contains more than 400 columns, dimensionality reduction was necessary.

- **Principal Component Analysis (PCA):**  
  - PCA was applied with a variance threshold of 0.95.
  - It was observed that only a few columns (features) were truly dependent on the target variable.
  - The remaining columns were removed, thereby reducing complexity and focusing on the most informative features.

## Model Training

Due to the limited dataset size (approximately 500 rows), machine learning algorithms were chosen over neural networks, which might underfit on small datasets.

<img src="images\image2.png" alt="best performing models" width="500" height="300">

1. **Experiment Tracking:**  
   - MLflow and DagsHub were used to log experiments, track model performance, and record hyperparameter tuning.

2. **Algorithm Comparison:**  
   - Various machine learning algorithms were evaluated.
   - **Gradient Boosting** emerged as the best-performing model for this dataset.
   
   <img src="images\image3.png" alt="best mse on different hyperparameters" width="500" height="300">

3. **Hyperparameter Tuning:**  
   - The model was further refined by tuning hyperparameters.
   - The best parameters were logged using MLflow for reproducibility and future reference.

## DVC Pipeline

- Created a robust pipeline for the model.
- If, after some time, we find that the model is not performing well on the given parameters:
  - We don't need to change the entire code.
  - Simply perform experimentation to obtain the best parameters.
- Update the `params.yaml` file.
- Run `dvc repro` in the terminal.
<img src="images\image4.png" alt="Training Pipeline" width="500" height="300">


## Key Findings and Suggestions for Improvement

- **Data Preprocessing:**  
  - While the data is initially preprocessed, it is recommended to provide the dataset with a broader range of possible preprocessing operations. This will offer more options for improvement, as it can be challenging to decide on the best preprocessing steps without multiple options.

- **Model Comparison – ML vs. Attention Mechanism:**  
  - Machine learning algorithms have outperformed attention-based (encoder-based) models on this dataset.
  - **Key reasons include:**
    - Attention-based models generally require a larger volume of data.
    - They benefit from greater data variety.
    - They typically need a larger number of training epochs to converge.
  - Therefore, for this dataset, simpler machine learning algorithms like Gradient Boosting are more effective.


## Tools and Frameworks

- **MLflow:** For tracking experiments and model training.
- **DagsHub:** For tracking URLs and version control.
- **DVC:** For pipeline management and data versioning.
- **FastAPI:** For building the backend API.
- **Streamlit:** For developing the frontend interface.

# To run the code and check for output run following commands
```
uvicorn app:app --reload
```
```
streamlit run streamlit_app.py
```


## Finally dockerize the file (but falied)
- make a docker file 'dockerfile'
- write all required code
- build a image with command 
```
docker build -t imago_assign .
```
- run the image ( move to container on docker dekstop ) command
```
docker run -p 5000:5000 imago_assign
```
- add a tag to push on docker hub
```
docker tag imago_assign alok8090/imago_assign:latest
```

- push to the docker hub commands
```
docker push alok8090/imago_assign:latest
```

- to pull and use the image run command 
```
docker pull alok8090/imago_assign:latest
docker run -p 5000:5000 alok8090/imago_assign:latest
```















## docker cmds
```
docker build -t imago_assignment .