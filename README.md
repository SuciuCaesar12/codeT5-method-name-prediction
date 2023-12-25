# Method Name Prediction using Transformers

## Table of Contents
#### [Introduction](#introduction)
#### [Dataset](#dataset)
#### [Methodology](#methodology)
#### [Results](#results)
#### [How to run](#how-to-run)
#### [Conclusions](#conclusions)

# Introduction <a name="introduction"></a>
* __Task Description__: Given the body and signature of a method written in Java, predict its name.
e.g. given the following method:

```java
public int ___(int a, int b) {
    return a + b;
}
```
the model should predict the name `add` for this method.

# Dataset <a name="dataset"></a>
* We extracted all the methods from [IntelliJ IDEA Community Edition](https://github.com/JetBrains/intellij-community) source code.
* This was done by iterating over all the files in the source code and extracting all the methods using regex patterns.
* Training set size: 102,629 methods (60%)
* Validation set size: 34,212 methods (20%)
* Test set size: 34,212 methods (20%)

__Note__: The validation set was used for hyperparameter tuning and the test set was used for final evaluation. For the final model we trained on the train and validation sets combined.

# Methodology <a name="methodology"></a>

We begin by leveraging the [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf) models and fine-tuning them on our dataset. Two distinct strategies have been employed for fine-tuning:

### Code Summarization:

- **Model Used:** [Salesforce/codet5p-220m-bimodal](https://huggingface.co/Salesforce/codet5-small-220M-bimodal)

- **Approach:**
    - The encoder-decoder model, as outlined in the paper, acts as a code-to-text summarization model.
    - Encoder takes the method body and signature as input.
    - Decoder input is `[TDEC] The name of the method is: `. `[TDEC]` indicates the start of text generation in the base model.
    - Target is the method name.
    - This strategy, referred to as `Code Summarization`, serves as a surrogate for `Method Name Prediction`. It involves conditioning the decoder's input on specific natural language text, focusing on predicting method names. The constrained input space is believed to retain knowledge acquired during pre-training, applicable to other code-to-text downstream tasks.

- **Example:**
    - Encoder Input: `public int </s>(int a, int b) { return a + b; }` (using `</s>` as the separator).
    - Decoder Input: `[TDEC] The name of the method is: `
    - Target: `add`

### Mask Prediction:

- **Model Used:** [Salesforce/codet5p-220m](https://huggingface.co/Salesforce/codet5p-220m)

- **Approach:**
    - Fine-tuning is performed on the model, which was originally trained with the objective of `Span Denoising`, involving masking spans of code and predicting the masked tokens.
    - Method name in the method body and signature is masked and used as input to the encoder.
    - Decoder predicts the masked tokens, which correspond to the method name.

- **Example:**
    - Input: `public int <extra_id_0>(int a, int b) { return a + b; }`
    - Target: `<extra_id_0> add </extra_id_1>`


# Results <a name="results"></a>

* **Evaluation Metrics:**
    * __Exact Match Accuracy__: The percentage of predictions that exactly match the target.
    * __ROUGE__: The average of the ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-LSUM scores. We assume the natural composition of method names and calculate ROUGE scores on the word level by splitting predicted and target names into words.

# How to Run <a name="how-to-run"></a>

### Environment Setup:

```shell
conda env create -f environment.yml
```

### Extract Data from IntelliJ IDEA Source Code:

1. **Download the source code:**
    - Obtain the IntelliJ IDEA source code from [here](https://github.com/JetBrains/intellij-community).

2. **Configure Data Extraction:**
    - Specify the path to the source code directory in the `notebooks_utils/extract_data.ipynb` file.

3. **Run Data Extraction Notebook:**
    - Execute the notebook; it will process the data and save it into three distinct files: `intellij-train.csv`, `intellij-val.csv`, and `intellij-test.csv`.

### Create Datasets:

1. **Dataset Creation Notebook:**
    - In each directory (`code_summarization` and `mask_prediction`) there is `create_dataset.ipynb` notebook.

2. **Provide Input:**
    - Specify the path to the .csv file generated in the previous data extraction step.

3. **Generate Datasets:**
    - Execute the notebook; it will create a `datasets.Dataset` object from the provided .csv file. 
    The resulting data will be saved into a .jsonl file. 
    Additionally, a .json file will be generated, containing information on how the dataset was created.
    Please refer to the notebook for more details on the dataset creation process and how it can be used for inference on a model.

### Fine-tune Models:

1. **Training Notebook**:
    - The notebook used for both strategies is `notebooks_utils/train.ipynb`. 
    - Make sure to set the task variable in the notebook to `mask-prediction` or `code-summarization` depending on the strategy you want to use.
    - Set `path_to_model` to the path of the model you want to fine-tune.
    - If `path_to_model` is set to `None`, the model will be downloaded from the HuggingFace model hub based on `checkpoint`

### Evaluate Models:

1. **Get Predictions**
    - To get predictions on some dataset, use the `notebooks_utils/inference.ipynb` notebook.
    - Similar to the training notebook, set the task variable to `mask-prediction` or `code-summarization` depending on the strategy the model was fine-tuned with.
    - Set `path_to_model` to the path of the model you want to make predictions with.
    - Set `path_to_dataset` to the path of the dataset you want to make predictions on.
    - The notebook will generate a .jsonl file containing the predictions.

2. **Evaluate Predictions**
    - To evaluate the predictions, use the `notebooks_utils/evaluate.ipynb` notebook.
    - Set `path_to_predictions` to the path of the .jsonl file containing the predictions.
    - Set `path_to_save_metrics` to the path of the .yaml file where the metrics will be saved.

# Conclusion <a name="conclusion"></a>

1. **Challenges in Method Name Prediction:**
    - Predicting the method name solely based on the method body and signature presents a challenging task, even for human understanding.
    - Valuable information may often be distributed across the class where the method is defined or within method calls embedded in the method body.
    - An ideal scenario for prediction is when the method functions independently, devoid of dependencies on other classes or methods, as seen in standalone algorithm implementations because the information is localized within the method body and signature.
    