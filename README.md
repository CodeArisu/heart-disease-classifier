
# Heart Disease Classifier

A Heart Disease probability classifier that classifies likelyhood of having heart diseases through various specified inputs. 

Where (P) > .0 == True otherwise (P) is False




## Preparation

Model: Logistic Regression

y = {1 if Class 1, 0 if Class 2}  
z = (∑i=1 n wi xi)+b
z = w⋅X+b

Independent observations: Each data point is assumed to be independent of the others means there should be no correlation or dependence between the input samples.

Preprocess: Standard Scaler

Xscaled = σX−μ

Target Class: Heart Disease

## Setup

## Initialize Python Environment

```sh
python -m venv .venv
```

## Activiate Environment

```sh
.venv\Scripts\Activate.bat
```

## Dependencies

```sh
pip install streamlit && pip install scikit-learn && pip install joblib
```
