# Indeed Machine Learning CodeSprint 

## About
Problem Description - [Tagging Raw Job Descriptions](https://www.hackerrank.com/contests/indeed-ml-codesprint-2017/challenges/tagging-raw-job-descriptions)

Challenge - Asks for a machine learning solution to accurately assign tags given the information in the job descriptions.

    
## Usage

Download the dataset [indeed_ml_dataset.zip](https://www.hackerrank.com/external_redirect?to=https://s3.amazonaws.com/hr-testcases-us-east-1/37111/assets/indeed_ml_dataset.zip) containing `train.tsv` and `test.tsv` files. Unzip and place it's contents into the `input/` folder
whose directory structure should look like:

    input 
      |---train.tsv 
      |---test.tsv
      
  
***Dependencies:***

    pip install -r requirements.txt

***Run:***

    cd Indeed-ML-codesprint-2017
    python source/model.py
    python source/generate_predictions.py
