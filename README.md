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
setting 'use_demo_data' = True will load the law_school.csv dataset along with an overview of this data and some questions one might ask if it were to be used to train a machine learning model.

### 2. Machnamh/demo_jupyter_notebooks/

### 3. Machnamh/demo_jupyter_notebooks/
This sample notebook has yet to be added.

[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)

