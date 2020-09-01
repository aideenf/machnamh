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


# Machnamh objectives
Fairness as a philosophy has no objective definition, and as such there is no consensus on a mathematical formulation for fairness. When training a Machine learning model to predict an outcome and hence influence decisions that will have positive or negative conseqnence for a person or group it is necessary to reflect on the worldview or philosophy of fairness that the model will reflect. The design and functionality of a machine learning model will likely reflect the worldview and values of those responsible for delivering the model. Machnamh may be used for reflecting upon the risk of introducing prejudice during the creation of a supervised machine learning applications for predictive modeling. The framework has been developed with a specific focus on those models which rank humans and supports either continuous numeric or binary predictions. Reflections prompted by this tool may require discussion, collaboration and agreement amongst various stakeholders within the business including relevant domain experts. Answers provided will form part of a report which will reflect the organization’s core values and worldview in relation to fairness. The report will provide a reference point for discussions between the producer and the consumer of the model with respect to the potential for these worldviews and values to be reflected in the models output. Would the historic decisions have differed, or would representation of a particular group be different in the data if discrimination was not occuring in the present, or had discrimination not occured in the past.

Worldview: In the context of this framework a "Worldview" is a set of assumptions about a physical and social reality pertaining to a human feature or attribute, or to the measurement of same. As context must be taken into consideration there is no one fundamentally correct worldview but rather a reflection of a particular philosophy of life, or a conception of the world, as it relates to each of an individuals' apparently quantifiable features or attributes. In the case of this framework, the focus is, in particular, on the worldview held concerning any disparities in features or attributes that might be detected across groups within protected features such as race, gender, age etc. A disparity may, for example, refer to a non-proportionate representation or a significant difference in distribution.

Two worldviews have been defined for this purpose: 
### Inherent or biological worldview: 
This worldview postulates that either chance or innate, inherent physiological, biochemical, neurological, cultural and/or genetic factors influence any disparities in features or attributes that might be detected across groups (categorised by race, gender, age etc). This worldview could be quite easily applied to the measurements of weight, height, BMI or similar easily quantifiable features to be used as predictors for a specific outcome. The worldview, however, becomes more complex for those human attributes or features which are harder to quantify, such as grit, determination, intelligence, cognitive ability, self-control, growth mindset, reasoning, imagination, reliability etc. This Inherent or biological worldview is closely aligned with the concept of individual fairness, where the fairness goal is to ensure that people who are ‘similar’ concerning a combination of the specific observable and measurable features or attributes deemed relevant to the task or capabilities at hand, should receive close or similar rankings and therefor achieve similar outcomes. With this worldview, observable and measurable features are considered to be inherently objective with no adjustments deemed necessary albeit with the knowledge that the human attributes or features considered critical to success may have been identified as such by the dominant group. Notwithstanding that a significant amount of the measurements used to gauge and/or measure these human features or attributes have been conceptualised, created or implemented by that same dominant group or that those historic outcomes may also have been influenced by prejudice towards a protected groups, or via favouritism towards the dominant group. This worldview might lead one to accept the idea that race, gender or class gaps are due to group shortcomings, not structural or systemic ones, and therefore the outcome “is what it is”, such that individuals should be ranked with no consideration to differences in outcome across groups. According to this worldview structural inequalities often perpetuated byracism, sexism and other prejudices are not considered to have any causal influence on outcomes. This worldview may also lead one to believe that representation of certain groups in specific fields (such as STEM) are disproportionate to the representation in the population due to inherently different preferences and/or abilities as opposed to the influence of social factors such as the exclusion, marginalisation, and undermining of the potential of the underrepresented group or to the favouritism (manifested through cognitive biases such as similarity bias etc) shown to other members of the dominant group. This worldview might lead one to conclude that certain groups of individuals do not avoid careers in certain sectors due to lack of mentorship or the existence of (or the perception of the existence of)an exclusionary workplace culture but rather because of their individual and inherent characteristics. 

### Social and environmental worldview: 
This worldview postulates that social and environmental factors, such as family income, parental educational backgrounds, school, peer group, workplace, community, environmental availability of nutrition, correct environment for sleep, stereotype threat(and other cognitive biases ) often perpetuated by racism, sexism and other prejudices have influenced outcomes in terms of any detected disparities across groups. Differences in outcome may be a reflection of inequalities in a society which has led to these outcome. Identifying this has important implications for the financial, professional, and social futures of particular protected groups within the population. Discrimination, privilege, institutional racism , sexism, ablism are examples of causal influences which may impact outcomes or representation. Disparities may have been caused by intentional,explicit discrimination against a protected group or by subtle, unconscious, automatic discrimination as the result of favoritism towards the reference group, or by other social and systemic factors. The term "affirmative action" is often used to justify the offering of opportunities to members of protected groups who do not otherwise appear to merit the opportunity. The offering of the opportunity is often based upon personal qualities that are usually hard to quantify in an entirely objective way. However it is important to note that due to social and environmental factors many measurements relating to human performance, merit, ability, etc are also not necessarily objective. 

*It is imperative to become familiar with these two worldview definitions before using the tool. Gaps in understanding of the ethical implications of poorly considered machine learning models which rank humans can have severe consequences for individuals as well as for groups and entire societies. 

