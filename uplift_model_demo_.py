import copy
import math

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from model import CausalChurnModel

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use("ggplot")

def plot_grouped_bar_chart(data, category):

    data = data.reset_index()

    category = category[-1]

    source = pd.melt(data, id_vars=[category])

    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=category, y="value", hue="variable", data=source)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    ax = plt.savefig('my_plot.png')

    st.pyplot(ax)

def validate_model_():

    st.markdown("Validation set segmented into matching and non-matching subsets using uplift model predictions.")
    st.markdown("1. Matching set comprises entries aligned with model's treatment suggestions.")
    st.markdown("2. Non-matching set consists of entries not aligned with model's treatment suggestions.")
    
    response = st.session_state.response
    validation_results = response["validation_response"]

    tMean = round(validation_results["Average Outcome"], 2)
    mtMean = round(validation_results["Average Matching Outcome"], 2)
    nmtMean = round(validation_results["Average Non Matching Outcome"], 2)

    tsdeltam =  mtMean - tMean
    tsdeltanm = round(nmtMean - tMean, 2)

    # plot_mean_sales_comparison_graph(validation_results)

    st.write("## The matching set typically possess lesser churn rate, due to better targetting.")

    col1, col2, col3 = st.columns([25, 25, 25])
        
    col1.metric("Population Mean", tMean, 
            help="Average churn rate of the entire validation set")

    col2.metric("Matching Set", mtMean, delta=tsdeltam, delta_color="normal", 
            help="Change in churn rate in the Matching Set")

    
    col3.metric("Non Matching Set", nmtMean, delta=tsdeltanm, delta_color="normal", 
            help="Change in churn rate in the Non Matching Set")

def visualize_individuals_():
    req_treatment_effects = st.session_state.req_treatment_effects
    configuration_meta = st.session_state.configuration_meta

    id_columns = configuration_meta["id_columns"]
    best_treatment_col = "best_predicted_treatment"

    all_columns = set(req_treatment_effects.columns)
    req_columns = all_columns - set([best_treatment_col])

    treatment_effects = req_treatment_effects[req_columns]
    treatment_effects = treatment_effects.groupby(id_columns).mean().reset_index(drop=True)
    treatment_effects += 0.50

    treatment_effects.set_index(req_treatment_effects[id_columns[-1]], inplace=True)

    plot_grouped_bar_chart(treatment_effects, id_columns)

    st.bar_chart(treatment_effects)

def segment_analysis(model, data):

    configuration_meta = st.session_state.configuration_meta
    to_drop = [configuration_meta["treatment_column"], configuration_meta["outcome_column"]]
    
    st.dataframe(data.drop(to_drop, axis=1), height=250)
    
    index_columns = set(configuration_meta["id_columns"])
    treatment_columns = set([configuration_meta["treatment_column"]])
    outcome_columns = set([configuration_meta["outcome_column"]])

    cat_columns =set(data.select_dtypes(include='object').columns)

    cat_columns -= index_columns
    cat_columns -= treatment_columns
    cat_columns -= outcome_columns

    cat_columns = ["None"] + list(cat_columns)
    segment = st.selectbox("Select Segment :", options=cat_columns)

    baction = st.button("Suggest Best Treatments")

    if baction:
        bte = model.find_uplift_of_a_segment_(data, segment)

        tab1, tab2 = st.tabs(["Tabular View", "Graphical View"])

        with tab1:
            st.markdown("Best Treatments")
            st.dataframe(bte)

        with tab2:
            st.markdown("Best Treatments")
            bte["base_action"] = 0
            bte += 0.50

            print("bte\n", bte)

            plot_grouped_bar_chart(bte, ["segment"])

            st.bar_chart(bte)

def find_best_treatment_(model, data):

    configuration_meta = st.session_state.configuration_meta
    to_drop = [configuration_meta["treatment_column"], configuration_meta["outcome_column"]]

    data_samples = data.sample(n=100).reset_index(drop=True)

    st.dataframe(data_samples.drop(to_drop, axis=1), height=250)

    selected_indices = st.multiselect('Select Individual Instances:'
    , data_samples.index)

    selected_rows = data_samples.loc[selected_indices]

    st.write('Selected Instances:')
    st.dataframe(selected_rows.drop(to_drop, axis=1))

    baction = st.button("Suggest Best Treatments")

    if baction:
        req_treatment_effects = model.find_best_treatment(selected_rows)
        st.session_state.req_treatment_effects = req_treatment_effects
        tab1, tab2 = st.tabs(["Tabular View", "Graphical View"])

        with tab1:
            st.markdown("Best Treatments")
            st.dataframe(req_treatment_effects)

        with tab2:
            st.markdown("Best Treatments")
            visualize_individuals_()

def train_model(data, id_columns, treatment_column
, control_column, outcome_column, objective):

    ccm = CausalChurnModel(data, id_columns=id_columns
    , treatment_column=treatment_column
    , outcome_column = outcome_column
    , control_column=control_column
    , objective=objective)

    response = ccm.train_uplift_model()

    return ccm, response

def configure_model_training_():

    path = ""  # Folder Path
    all_data = ["None", "simple_churn_data", "crm_churn_data"] # dataset names

    data = st.selectbox("Select Data", all_data)

    if data != "None":
        data = pd.read_csv(f"{path}{data}.csv")
        st.markdown("#### Selected Data")
        st.dataframe(data, height=250)
        all_columns = list(data.columns)

    else:
        all_columns = []
        available_treatments = []

    st.markdown("Configure Meta Information")

    all_columns = ["None"] + all_columns
    id_columns = st.multiselect("Select ID Columns :- Unique identifiers for each record", options=all_columns)
    treatment_column = st.selectbox("Select Treatment Column :- Indicators of the intervention or action applied to each record", options=all_columns)
    
    control_value = ""

    if treatment_column != "None":
        available_treatments = list(data[treatment_column].unique())
        available_treatments = ["None"] + available_treatments

        control_value = st.selectbox("Select Control Group :-  A subset of records that did not receive the intervention", options=available_treatments)

    outcome_column = st.selectbox("Select Outcome Column :- The result or response variable of interest", options=all_columns)

    objective = st.selectbox("Objective :- Specifies whether the goal of the uplift model is to maximize or minimize the outcome", ["None", "maximize", "minimize"])
    train = st.button("Train Uplift Model")

    model = ''
    response = ''

    configuration_meta = dict()

    if train:
        model, response = train_model(data
        , id_columns=id_columns
        , treatment_column=treatment_column
        , outcome_column=outcome_column
        , control_column=control_value
        , objective=objective)

        st.markdown("### Training Response")
        st.json(response)

    configuration_meta["id_columns"] = id_columns
    configuration_meta["treatment_column"] = treatment_column
    configuration_meta["control_column"] = control_value
    configuration_meta["outcome_column"] = outcome_column
    configuration_meta["objective"] = objective

    return model, response, data, configuration_meta

st.sidebar.title("RetentionAI - Boost Customer Retention with Causal AI")
st.sidebar.text("Leverage AI to tailor \nretention strategies")

option = st.sidebar.radio("Pages", options=["Model Configuration", "Individual Analysis", "Segment Analysis", "Model Validation"])

if option == "Model Configuration":
    st.markdown('## Model Configuration')
    st.markdown('Prepare and transform data to create relevant features for the model.')

    with st.spinner("Please Wait.."):
        model, response, data, configuration_meta = configure_model_training_()
    
        st.session_state.model = model
        st.session_state.response = response
        st.session_state.data = data
        st.session_state.configuration_meta = configuration_meta

if option == "Individual Analysis":

    st.markdown('## Individual Analysis')
    st.markdown("Analyze each action's impact on individual churn probability to make informed decisions.")
    
    model = st.session_state.model
    data = st.session_state.data

    best_treatment_ = find_best_treatment_(model, data)

if option == "Segment Analysis":
    st.markdown('## Segment Analysis')
    st.markdown('Analyze how each action would affect segment-specific churn probabilities to make informed decisions.')

    model = st.session_state.model
    data = st.session_state.data

    best_treatment_ = segment_analysis(model, data)

if option == "Model Validation":
    st.markdown('## Model Validation')
    validate_model_()
