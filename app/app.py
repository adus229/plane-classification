import os
import pathlib
from pickle import NONE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from PIL import Image
import yaml 


#Load constantes
with open("app.yaml",'r') as config_file:
    config = yaml.safe_load(config_file)
    IMAGE_WIDTH = config["image_width"]
    IMAGE_HEIGHT = config["image_height"]
    IMAGE_DEPTH = config["image_depth"]
    NEURAL_NETWORKS_DIR_NAME= config["neural_networks_dir_name"]
    SVM_DIR_NAME= config["svm_dir_name"]
    TRANSERT_LEARNING_DIR_NAME=config["transfert_learning_dir_name"]
    MODELS_DIR = pathlib.Path(config["models_dir"])
    CLASSES_FILE_DIR = pathlib.Path(config["classes_file_dir"])



# Functions

def load_image(path):
    """Load an image as numpy array
    parameters:
    -----------
    path: to the image
    """
    return plt.imread(path)
    

def predict_image(path, model,type_model,classes_names=NONE):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction 
    classes_names: list of classes of avions to predict
    
    Returns
    -------
    Predicted class
    """
    
    images = np.array([np.array(Image.open(path).resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT)
            ))])

    if type_model == NEURAL_NETWORKS_DIR_NAME or type_model==TRANSERT_LEARNING_DIR_NAME:
        _prediction_vector = model.predict(images)
        _prediction_vector= _prediction_vector.reshape(_prediction_vector.shape[1])
        _predicted_classe = np.argmax(_prediction_vector)
        _prob= round(np.max(_prediction_vector),2)

        
    else:
        _img_to_predict = np.array([images.flatten()])
        _predicted_classe = model.predict(_img_to_predict).astype(int)[0]
        _prediction_vector = model.predict_proba(_img_to_predict).flatten()
        _prob = _prediction_vector[_predicted_classe]

    #Check if the classes name has been given
    if classes_names is not None : 
            _predicted_classe = classes_names[_predicted_classe]

    return {
            "classe": _predicted_classe,
            "classe_prob": _prob,
            "all_prob": _prediction_vector
        } 

@st.cache()
def load_model(path,type_model):
    """Load tf/Keras model for prediction
    """
    if type_model==NEURAL_NETWORKS_DIR_NAME or type_model==TRANSERT_LEARNING_DIR_NAME:
        return tf.keras.models.load_model(path)
    else :
        with open(path,"rb") as file:
            return pickle.load(file)
   
#Header of the application
st.set_page_config(layout="wide")
col1,col2=st.columns([1,8])
with col1:
    st.image("./logo_plane.png")
with col2:
    st.title("Identification d'avion")


#The sidebar of options
with st.sidebar:
    #Use model directory as value af the selected box
    type_models = st.radio("Type de modèle",
                    (NEURAL_NETWORKS_DIR_NAME,SVM_DIR_NAME,TRANSERT_LEARNING_DIR_NAME),
                    format_func= lambda option: "Réseaux de neurone" 
                            if option== NEURAL_NETWORKS_DIR_NAME
                            else "SVM" if option == SVM_DIR_NAME
                            else "Transfert learning" 
    )

    models_path = MODELS_DIR/type_models
    models_names = os.listdir(models_path) 
    model_name_box = st.selectbox(
                    "Sélectionner le modèle",
                    models_names,
                    format_func= lambda p: os.path.splitext(p)[0]
    )

    show_prob_bar = st.checkbox("Afficher les probabilités",1)

    type_graph = st.radio("Choisir le type de graphique",
        ("barre","surface"),disabled=(show_prob_bar == 0 )
    )


uploaded_file = st.file_uploader("Charger une image d'avion") 

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)

#The button is enabled only if a picture and a model has been choosen
predict_btn = st.button("Identifier", disabled=( uploaded_file is None or model_name_box is None))

if predict_btn:
    
    model_selected_path = MODELS_DIR/type_models/model_name_box
    model = load_model(model_selected_path,type_models)
    
    with open(CLASSES_FILE_DIR/(pathlib.Path(model_name_box).stem+".yaml"),"r") as classes_file:
        classes_names =yaml.safe_load(classes_file)
    
    pred_results = predict_image(uploaded_file, model,type_models,classes_names)
    choix = pathlib.Path(model_name_box).stem.capitalize()

    st.title("Résultats de prédiction")
    res_col1,res_col2 = st.columns(2)
    res_col1.markdown(f"**{choix}**: {pred_results['classe']}")
    res_col2.markdown(f"**Probabilité**: {float(pred_results['classe_prob'])}")

    #Show prob graph
    if show_prob_bar:
        chart_df = pd.DataFrame( pred_results["all_prob"],
                index=classes_names,
                columns=["Probabilité"])
        if type_graph == "barre":
            st.bar_chart(chart_df)
        else:
            st.area_chart(chart_df)
    st.balloons()

