# Description
This project is mean to classify plane from pictures.
 


# Description of main files
Here are the contents of each notebook
- [notebooks/config.yaml](notebooks/config.yaml): configuration file to train and create the models
- [app/config.yaml](app/config.yaml): configuration file to run the streamlit application
- [data_download.ipynb](./notebooks/data_download.ipynb): is to download the database used, and place them in the appropriate code 
- [train_classification_model.ipynb](notebooks/train_classification_model.ipynb): it is a draft of the model
- [cleaned_train_classification_model.ipynb](notebooks/cleaned_train_classification_model.ipynb): is a cleaned version of the draft
- [classification_neural_network_with_target](notebooks/classification_neural_network_with_target.ipynb): is the notebook of neural network with target
- [classification_svm_with_target](notebooks/classification_svm_with_target.ipynb):the notebook for the model of SVM, with a polynomial kernel density
- [classification_transfert_learning_with_target](notebooks/classification_transfert_learning_with_target.ipynb): a notebook of a transfert learning model 

# Installation
- clone the project
- create new virtual env( not mandatory but recommanded)
- install all the requirement dependencies from the requirement file
    ```pip install -r requirements.txt```
# Run
After installation, it's important to setup the appropriate value in the configuration files before running

# Models 
## Saved model
Cause of a bandwith problem the saved models are not there currently but they are available [here](https://drive.google.com/drive/folders/1G7lrqa0cS42Rr722_QC-WErRDa6WL9R-?usp=sharing).

## Publication 
A first version is published here, and for the same reason it not yet take into account the models object. But you can clone the model and try it locally. 
The first version is at the following: 

# Exemples

![With probabilities in bar](examples/ex1.png)
![With probabilities in surface](examples/ex2.png)

