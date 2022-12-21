import streamlit as st
import pickle
import pandas as pd
from random import choice

# ---------------Page Setup-----------------#

st.set_page_config(layout="wide", page_title="Cardiac-Arrest-Detector")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set Title
st.title("Cardiac Arrest Prediction")
st.markdown("#### Using Artificial Intelligence")
st.markdown("#")
st.markdown("#")

col1, col2 = st.columns([2, 5])
with col1:
    col1.image("heart.jpg")

with col2:
    col2.write(
        "Cardiac arrest is a sudden, unexpected loss of heart function, breathing and consciousness.  \
        Cardiac arrest usually results from an electrical disturbance in the heart. It's not the same as a heart attack. \
        The main symptom is loss of consciousness and unresponsiveness. This medical emergency needs immediate CPR or \
        use of a defibrillator."
    )


# --------------- Slider Section-----------------------------
st.sidebar.markdown("# Select AI Model")
model_name = st.sidebar.radio(
    "Algorithms", ["Random Forest", "XGBoost", "Gradient Boosting"]
)
load_model_button = st.sidebar.button("Load Algorithm")
# st.sidebar.markdown("---")
st.sidebar.markdown(f" # Dataset Infomation")
st.sidebar.write("Description of the dataset's features are given below")

with st.sidebar.expander("Read more about dataset features"):
    st.markdown(
        ":one: **Age:** Age of the patient [years]  \n"
        ":two: **Sex:** Sex of the patient [M: Male, F: Female]  \n"
        ":three: **ChestPainType:** Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]  \n"
        ":four: **RestingBP:** Resting blood pressure [mm Hg]  \n"
        ":five: **Cholesterol:** Serum cholesterol [mm/dl]  \n"
        ":six: **FastingBS:** Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]  \n"
        ":seven: **RestingECG:** Resting electrocardiogram results [Normal: Normal, ST:\
        having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), \
        LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]  \n"
        ":eight: **MaxHR:** Maximum heart rate achieved [Numeric value between 60 and 202]  \n"
        ":nine: **ExerciseAngina:** Exercise-induced angina [Y: Yes, N: No]  \n"
        ":keycap_ten: **Oldpeak:** Oldpeak = ST [Numeric value measured in depression]  \n"
        ":one::one: **ST_Slope:** The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]  \n"
        ":one::two:**Cardiac Arrest**: Output class [1: cardiac arrest, 0: Normal]"
    )
st.markdown("---")


# Fuction for loading and caching the model
@st.cache(allow_output_mutation=True)
def load_data(name):
    if name == "Gradient Boosting":
        model = pickle.load(open("gradient_model.pkl", "rb"))
    elif name == "Random Forest":
        model = pickle.load(open("RF_model.pkl", "rb"))
    elif name == "XGBoost":
        model = pickle.load(open("Xgb_model.pkl", "rb"))
    return model


# Load model
if load_model_button:
    model = load_data(model_name)
    if "model" not in st.session_state:
        st.session_state["model"] = model


##### FEATURES ######

chestTypeMap = {"TA": 3, "ATA": 1, "NAP": 2, "ASY": 0}
sexMap = {"Male": 1, "Female": 0}
StMap = {"UP": 2, "FLAT": 1, "DOWN": 0}
ExerciseMap = {"NO": 0, "YES": 1}
restingEcgMap = {"Normal": 1, "ST": 2, "LVH": 0}

st.markdown("#")
st.subheader("Input Patient's Record")
col2, col3 = st.columns(2)
with col2:
    Age = col2.text_input("Age", placeholder="Enter Age")
    Sex = sexMap[col2.selectbox("Sex", ["Male", "Female"])]
    ChestPain = chestTypeMap[
        col2.selectbox("ChestPainType", ["TA", "ATA", "NAP", "ASY"])
    ]
    RestingBP = col2.slider("RestingBp", 0, 200, 1)
    Chol = col2.slider("Cholesterol", 0, 603, 1)

with col3:
    FastingBS = col3.selectbox("FastingBp", [0, 1])
    RestingECG = restingEcgMap[col3.selectbox("RestingECG", ["Normal", "ST", "LVH"])]
    ExerciseAngina = ExerciseMap[col3.selectbox("ExcerciseAngina", ["NO", "YES"])]
    MaxHR = col3.slider("MaxHR", 60, 202, 1)
    Oldpeak = col3.slider("OldPeak", -2.6, 6.2, 1.0)
ST_slope = StMap[st.selectbox("ST_Slope", ["UP", "FLAT", "DOWN"])]

st.markdown("#")
predict = st.button("Predict")


# #----------------Dataset Section---------------#
input_data = {
    "Age": Age,
    "Sex": Sex,
    "ChestPainType": ChestPain,
    "RestingBP": RestingBP,
    "Cholesterol": Chol,
    "FastingBS": FastingBS,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "Oldpeak": Oldpeak,
    "ST_Slope": ST_slope,
}

#     # Create input data
input_features = pd.DataFrame(input_data, index=[0])

# st.dataframe(input_features)


# -----------------Prediction-----------------------

st.markdown("AI's Prediction")

medication_recommendation = [
    "Epinephrine. This medication can help to increase blood flow to the heart and brain. It should be administered every 3 to \
    5 minutes early in CPR (every other cycle) for asystole, ventricular fibrillation, ",
    "Amiodarone. This medication can help to stabilize the heart rhythm.",
    "Lidocaine.This medication can help to reduce abnormal electrical activity in the heart.",
    "Atropine.This medication can help to increase the heart rate in certain types of cardiac arrest.",
]


try:
    if predict:
        predictions = st.session_state.model.predict(input_features)
        if predictions == 0:
            st.success(
                "AI predicts there's no evidence of cardiac arrest.Congratulations!!! "
            )
        elif predictions == 1:
            st.error(
                choice(
                    [
                        f"AI predicts there's evidence of cardiac arrest and recommends that you take {choice(medication_recommendation)}",
                        "It seems you may have a cardiac arrest. The primary treatment for cardiac arrest is cardiopulmonary resuscitation (CPR)\
                     and defibrillation.CPR involves chest compressions and rescue breathing to\
                     help circulate blood and oxygen to the body, while defibrillation involves using a\
                    device to deliver an electric shock to the heart to try to restore a normal heart rhythm. Perhaps you should try out one of these exercises",
                        "There's evidence of cardiac arrest. If CPR and defibrillation are not successful in restoring a normal heart rhythm,\
                         medications may be used to try to stabilize the heart. Some common medications used to treat cardiac arrest include: \
                        (1)  Vasopressin, as an alternative to epinephrine every 3 to 5 minutes (every second BLS cycle) for asystole, bradycardia, PEA. \
                        (2) Sodium bicarbonate \
                        (3) Amiodarone",
                    ]
                )
            )
        else:
            st.empty()
except ValueError:
    st.error("Please ensure to fill in all required fields!")
