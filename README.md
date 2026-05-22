# Multimodal Skin Lesion Classification

## Project Overview
Develop a machine learning pipeline to classify skin lesions as requiring medical attention (BCC, MEL, SCC, ACK) or harmless (NEV, SEK). The task requires integrating cellphone images with tabular clinical features to execute a binary classification task.

Images can be found at the following Google Drive links:

* <https://drive.google.com/file/d/1JTvMkEaR3AxqAagYzfLebnS_RavA7agR/view?usp=drive_link>
* <https://drive.google.com/file/d/1bQxyUXwvUn2cPjb5gIp-jB3CUpqD4Ca1/view?usp=drive_link>


## Data citation

This dataset comes from work from a research group working with the Dermatological and Surgical Assistance Program (PAD) of the Federal University of Espírito Santo, a nonprofit program that provides free skin lesion treatment, in particular, to low-income people who cannot afford private treatment. The data is released under a CC BY 4.0 license. See also the publication describing the dataset:

<blockquote style="margin-left: 5em">
<p>
  <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/">
PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones
  </a>
  <br />
Andre GC Pacheco et. al.
  <br />
August 2020
  <br />
Data Brief. 2020 Aug 25;32:106221. doi: 10.1016/j.dib.2020.106221
  <br />
</p>
</blockquote>

## Model Progression

### Problem 1: Random Forest Pipeline
1. **Feature Integration:** The pipeline fuses 21 clinical metadata features (one-hot encoded for categorical variables) with manually engineered image features. Custom image metrics include regional image contrast and RGB color channel variations (calculated via standard deviation).

2. **Cross-Validation Strategy:** To ensure robust evaluation and prevent data leakage (since some patients have multiple lesions), the model is validated using a Stratified Group 5-Fold Cross-Validation (StratifiedGroupKFold). Folds are strictly grouped by patient_id while maintaining the minority class ratios.

3. **Hyperparameter Tuning:** The pipeline utilizes GridSearchCV to exhaustively search for the optimal model complexity, targeting the highest AUROC score. The final best-performing model utilizes:

$\verb|n_estimators=500|$ (500 decision trees)

$\verb|max_depth=16|$

$\verb|min_samples_leaf=5|$

$\verb|class_weight='balanced'|$ (to penalize majority class misclassifications)
