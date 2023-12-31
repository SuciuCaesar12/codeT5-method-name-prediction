# Method Name Prediction using Transformers

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Conclusions](#conclusions)

# Introduction <a name="introduction"></a>

* __Summary__: This repository contains the code for the test task given in __Jetbrains Application Internship 2023__.

* __Task Description__: Given the body and signature of a method written in Java, predict its name.
e.g. given the following method:

```java
public int _(int a, int b) {
    return a + b;
}
```
the model should predict the name `add` for this method.

# Dataset <a name="dataset"></a>
* We extracted all the methods from [IntelliJ IDEA Community Edition](https://github.com/JetBrains/intellij-community) source code.
* This was done by iterating over all the files in the source code and extracting all the methods using regex patterns.
  * Training set size: 102K methods (60%)
  * Validation set size: 34K methods (20%)
  * Test set size: 34K methods (20%)

__Note__: The validation set was used for hyperparameter tuning and the test set was used for final evaluation. For the final model we trained on the train and validation sets combined.

# Methodology <a name="methodology"></a>

We begin by leveraging the [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/pdf/2305.07922.pdf) models and fine-tuning them on our dataset. Two distinct strategies have been employed for fine-tuning:

### Code Summarization:

- **Model Used:** [Salesforce/codet5p-220m-bimodal](https://huggingface.co/Salesforce/codet5-small-220M-bimodal)

- **Approach:**
    - The encoder-decoder model, as outlined in the paper, can act as a code-to-text summarization model.
    - Encoder takes the method body and signature as input.
    - Decoder takes as input `[TDEC] The name of the method is: `. `[TDEC]` indicates for the decoder the start of text generation.
    - Target is the method name.
    - This strategy, referred to as `Code Summarization`, serves as a surrogate objective for `Method Name Prediction`. It involves conditioning the decoder's input on specific natural language text, focusing on predicting method names.

- **Example:**
    - Encoder Input: `public int </s>(int a, int b) { return a + b; }` (`</s>` special separator token)
    - Decoder Input: `[TDEC] The name of the method is: `
    - Target: `add`

### Mask Prediction:

- **Model Used:** [Salesforce/codet5p-220m](https://huggingface.co/Salesforce/codet5p-220m)

- **Approach:**
    - For fine-tuning the model we sue `Span Denoising` objective as outlined in the paper.
    - We take the whole method and mask its name and then train the model to predict the masked name.

- **Example:**
    - Input: `public int <extra_id_0>(int a, int b) { return a + b; }`
    - Target: `<extra_id_0> add </extra_id_1>`


# Results <a name="results"></a>

* **Evaluation Metrics:**
    * __Exact Match Accuracy__: The percentage of predictions that exactly match the target.
    * __ROUGE__: We assume the natural composition of method names and calculate ROUGE scores on the word level by splitting predicted and target names into words.
    The columns represent the range of number of lines in the body.
    
    * __Code Summarization__:
  
      | Metric      | 0-5   | 5-10  | 10-20 | 20-50 |
      |-------------|-------|-------|-------|-------|
      | exact_match | 0.441 | 0.358 | 0.235 | 0.179 |
      | rouge1      | 0.669 | 0.594 | 0.515 | 0.462 |
      | rouge2      | 0.433 | 0.287 | 0.215 | 0.136 |
      | rougeL      | 0.668 | 0.593 | 0.516 | 0.464 |
      | rougeLsum   | 0.668 | 0.593 | 0.515 | 0.462 |

    * __Mask Prediction__:

      | Metric      | 0-5   | 5-10  | 10-20 | 20-50 |
      |-------------|-------|-------|-------|-------|
      | exact_match | 0.443 | 0.311 | 0.260 | 0.194 |
      | rouge1      | 0.663 | 0.580 | 0.575 | 0.577 |
      | rouge2      | 0.418 | 0.263 | 0.257 | 0.210 |
      | rougeL      | 0.662 | 0.578 | 0.573 | 0.577 |
      | rougeLsum   | 0.662 | 0.578 | 0.574 | 0.575 |

    * The results indicate that `code summarization` performs better than `mask prediction` on shorter methods, while `mask prediction` performs better on longer methods.

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
    - Execute the notebook; it will iterate through all `.java` files in the source code directory and extract all methods using regex patterns.
    - The data will be saved in three `.csv` files: `intellij-{train,valid,test}.csv` with 60%, 20%, 20% splits respectively.
    - The files will contain 2 columns: 
      - `code`: the entire method with its name, too.
      - `name`: contains the method name.

### Create Datasets:

1. **Dataset Creation Notebooks:**
    - In each directory (`code_summarization` and `mask_prediction`) there is `create_dataset.ipynb` notebook.

2. **Provide Input:**
    - Specify the path to the .csv file generated in the previous data extraction step.

3. **Generate Datasets:**
    - Execute the notebook; it will create a `datasets.Dataset` object from the provided .csv file. 
    - The resulting data will be saved into a `.jsonl` file with the following features:
      - __Code Summarization__:
        - `input_ids`: sequence of token ids for the encoder
        - `attention_mask`: attention mask for the encoder
        - `decoder_input_ids`: sequence of token ids for the decoder
        - `decoder_attention_mask`: attention mask for the decoder
        - `labels`: sequence of token ids for the decoder target

      - __Mask Prediction__:
        - `input_ids`: sequence of token ids for the encoder
        - `attention_mask`: attention mask for the encoder
        - `labels`: sequence of token ids for the decoder target

    - Additionally, a `.json` file will be generated, containing information on how the dataset was created.

    __Note__: For computational reasons, the dataset was already tokenized and the length of the sequences was determined by the longest sequence in the batch. 
    During any inference on a model, the dataset should not be shuffled and the batch size chosen should evenly divide the batch size used for creating the dataset to ensure that examples batched together have the same length.

### Fine-tune Models:

1. **Training Notebook**:
    - The notebook used for both strategies is `notebooks_utils/train.ipynb`. 
    - Make sure to set the task variable in the notebook to `mask-prediction` or `code-summarization` depending on the strategy you want to use.
    - Set `path_to_model` to the path of the model you want to fine-tune.
    - If `path_to_model` is set to `None`, the model will be downloaded from the HuggingFace model hub based on `checkpoint` and `task` variables.

### Evaluate Models:

1. **Get Predictions**
    - To get predictions on a dataset, use the `notebooks_utils/inference.ipynb` notebook.
    - Similar to the training notebook, set the task variable to `mask-prediction` or `code-summarization` depending on the strategy the model was fine-tuned with.
    - Set `path_to_model` to the path of the model you want to make predictions with.
    - Set `path_to_dataset` to the path of the dataset you want to make predictions on.
    - The notebook will generate a `.jsonl` file with the following features:
      - `input_code`: the input code (text)
      - `labels`: the method name (text)
      - `prediction`: the predicted method name (text)

2. **Evaluate Predictions**
    - To evaluate the predictions, use the `notebooks_utils/evaluate.ipynb` notebook.
    - Set `path_to_predictions` to the path of the .jsonl file containing the predictions.
    - Set `path_to_save_metrics` to the folder where you want to save the evaluation metrics.
    - The notebook will create a `metrics.yml` file containing the evaluation metrics. for different ranges of method lengths.

# Conclusion <a name="conclusion"></a>

1. **Challenges in Method Name Prediction:**
    - Predicting the method name solely based on the method body and signature presents a challenging task, even for human understanding.
    - Valuable information may often be distributed across the class where the method is defined or within method calls embedded in the method body.
    - An ideal scenario for prediction is when the method functions independently, devoid of dependencies on other classes or methods, as seen in standalone algorithm implementations because the information is localized within the method body and signature.
    - However, the ideal case doesn't occur often in practice, and the method body and signature are often insufficient to predict the method name.

2. **Future improvements:**
    - Incorporate information regarding the class or the structure of the project to improve the prediction.
    - Explore other data sources to fine-tune the model on.
