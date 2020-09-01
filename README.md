# Machnamh installation via pip
Install alpha version from pipy
https://test.pypi.org/project/machnamh-unmakingyou/

Using command:
pip install -i https://test.pypi.org/simple/ machnamh-unmakingyou

or for a specific release number use:
pip install -i https://test.pypi.org/simple/ machnamh-unmakingyou==0.0.10

# Using the library
See the Machnamh/demo_jupyter_notebooks/ folder for sample code which utilises the library for 3 steps in fairness review process.

### 1. Machnamh/demo_jupyter_notebooks/machnamh_step_one_review_prepare_data.ipynb
This notebook (or the following code), can be used to invoke the Machnamh data review UX
```
import machnamh 
from machnamh import pre_process as mpp
dpUI = mpp.data_pre_process_UI()
dpUI.render(use_demo_data = True)
```
Setting ```use_demo_data = True``` will load the law_school.csv dataset along with an overview of this data and some questions one might ask if this data were to be considered for the purpose of training a machine learning model

### 2. Machnamh/demo_jupyter_notebooks/machnamh_step_two_train_model_analyse_output.ipynb
This notebook shows:
1. Reloading the transformed data and data_summary created in step 1 using Machnamh functions
2. Creating a Machine learning model of your choice with the transformed data, in the sample we have used Logistic Regression which is then ranked/ordered according to the probability of having an output class of 1. 
3. Using Machnamh function to invoke a UX which uses the Aequitas API to audit fairness across protected groups
4. Using Machnamh function to invoke a UX which uses the SHAP library for explainability. 
5. Save the ranked list so that it may be processed in sample notebook 3.

### 3. Machnamh/demo_jupyter_notebooks/
This sample notebook has yet to be added.

[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)

