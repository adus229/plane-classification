from utils.load_image import load_image
import streamlit as st
from config import Config, Paths
from models.registry import get_model_class, get_model_label, available_models
from planeclf.data.pipeline import load_and_resize
import yaml
from planeclf.inference.predictor import predict_topk
import pandas as pd

import altair as alt

config = Config.get()
paths = Paths.get()
st.session_state.uploaded_file = None
st.session_state.loaded_image = None
st.session_state.predictions = [[None, 0.0]]


def show_chart(df: pd.DataFrame, target: str = "manufacturer"):
    # remettre l'index en colonne si besoin
    if target not in df.columns and df.index.name == target:
        df = df.reset_index()

    # tri croissant par proba
    df = df.sort_values("Probability", ascending=False).copy()
    df["Probability"] = df["Probability"].astype(float)

    st.header("Prediction probabilities")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{target}:N",
                    # soit on laisse l'ordre des données:
                    sort=None,
                    # ou on force par la colonne Probability:
                    # sort=alt.SortField(field="Probability", order="ascending"),
                    title=target.capitalize()),
            x=alt.X("Probability:Q", title="Probability", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip(f"{target}:N", title=target.capitalize()),
                     alt.Tooltip("Probability:Q", format=".2%")],
        )
    )
    st.altair_chart(chart, use_container_width=True)
st.logo(paths.assets_dir/"logo_plane.png")
st.set_page_config(page_title="Plane Classifier", layout="wide")


# BUILD THE SIDEBAR
with st.sidebar:

    characteristics = ["manufacturer", "family", "variant"]

    # The type of thing to classify
    target = st.selectbox(
        "What do you want to find ?",
        characteristics,

    )

    models_dict = {name: get_model_label(name) for name in available_models
                   }
    # Use model directory as value af the selected box
    model_selected = st.radio("Type of model",
                              options=(models_dict.keys()),
                              format_func=lambda option: models_dict[option],
                              )

    show_prob_bar = st.checkbox("Show probabilities", 1)

    # type_graph = st.radio("Type of graph", ("barre", "surface"), disabled=(show_prob_bar == 0))
    
    model = get_model_class(model_selected)()

    model = model.load(
        f"{config.paths.models_dir}/{target}/{model_selected}/{model_selected}")
    
            # Load class names
    with open(config.paths.models_dir/target/model_selected/"classes.yaml", "r", encoding="utf8") as classes_file:
        classes_names = yaml.safe_load(classes_file)

    slider_options = list(range(1, len(classes_names) + 1))

    topK = st.select_slider("Top K predictions", options=slider_options, value=3)

main_col,side_col = st.columns([2,2])

#build the main column
with main_col:
    img_col, title_col = st.columns([1, 8])
    with img_col:
        st.image(paths.assets_dir/"logo_plane.png")
    with title_col:
        st.title("Plane Classifier ✈️")

    col_file1, col_result = st.columns(2)
    with col_file1:
        st.session_state.uploaded_file = st.file_uploader(
            "Upload an image of a plane", type=["jpg", "jpeg", "png"])
        if st.session_state.uploaded_file:
            st.session_state.loaded_image = load_image(
                st.session_state.uploaded_file)

    if st.session_state.uploaded_file:
        resized_image = load_and_resize(
            [st.session_state.uploaded_file], size=(config.img_width, config.img_height))

        st.session_state.predictions = predict_topk(model, resized_image, k=topK, classes=classes_names)
        st.session_state.predictions = st.session_state.predictions[0]
        chart_df = pd.DataFrame(st.session_state.predictions,columns=[target, "Probability"])
        chart_df = chart_df.set_index(target)
        chart_df = chart_df.sort_values(by="Probability", ascending=False)

        if show_prob_bar and  st.session_state.uploaded_file:
            show_chart(df=chart_df, target=target)


#Build side column
with side_col:
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.header("Image preview")
    with st.container( border=False, ):
        st.image(paths.assets_dir /
             "coming-soon.png" 
             if st.session_state.loaded_image is None else st.session_state.loaded_image, 
             width="stretch", 
             output_format="auto")



with col_result:
    left, center, right = st.columns([1, 2, 1]) 
    with center:
        st.header("Prediction")
        st.metric(
            label=target.capitalize(),
            value=st.session_state.predictions[0][0],
            delta=f"{st.session_state.predictions[0][1]*100:.2f}%"
        )

