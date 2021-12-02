import base64
from io import BytesIO

import streamlit as st
import os
import streamlit.components.v1 as stc


import pandas as pd
import numpy as np


import plotly.express as px


import matplotlib

matplotlib.use('Agg')  # To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib


def main():
    st.title("Medictionary Disease Predictor Web App")

    menu = ["Predictor Tool", "Dataset Information", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    #Loading our Dataset
    df = pd.read_csv("Training_Disease.csv")

    if choice == "Dataset Information":
        st.subheader("Dataset Information")

        if st.checkbox("Show DataSet"):
            number = st.number_input("Number of Rows to View", min_value=1, max_value=10, value=5, step=1)
            st.dataframe(df.head(number))

        if st.button("Column Names"):
            st.write(df.columns)

        if st.button("Statistical Information about the dataset"):
            s_df_des = df.describe()
            st.write(s_df_des)

        if st.checkbox("Shape of Dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension by", ("Rows", "Columns"))
            if data_dim == 'Rows':
                st.text("Number of  Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])

        if st.checkbox("Select Columns To Show"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect('Select', all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.button("Data Type"):
            st.markdown('**int64** and **object**')

        if st.button("Value Counts"):
            st.text("Value Counts By Class")
            st.write(df.iloc[:, -1].value_counts())

        st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.checkbox("Scatter Plot for Feature Name vs Prognosis"):
            x_axis_options = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
                              'chills',
                              'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                              'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
                              'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
                              'lethargy',
                              'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
                              'breathlessness',
                              'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine',
                              'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
                              'abdominal_pain', 'diarrhoea',
                              'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
                              'fluid_overload',
                              'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
                              'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
                              'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
                              'pain_during_bowel_movements',
                              'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
                              'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
                              'puffy_face_and_eyes',
                              'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
                              'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
                              'hip_joint_pain', 'muscle_weakness',
                              'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
                              'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
                              'bladder_discomfort',
                              'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
                              'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
                              'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches',
                              'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                              'rusty_sputum',
                              'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
                              'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                              'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
                              'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
                              'blackheads', 'scurring', 'skin_peeling',
                              'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
                              'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis']
            x_axis = st.selectbox('Which feature do you want to explore?', x_axis_options)
            fig = px.scatter(df,
                             x=x_axis,
                             y='prognosis',
                             title=f'Prognosis vs. {x_axis}')

            st.plotly_chart(fig)

        if st.checkbox("Subplots for count of symptoms"):
            for i in df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                bar = df.groupby(i).size().plot(kind='bar', ax=ax)
                plt.xticks(rotation=0)
                fig.suptitle("Count of Symptom \"" + i + "\"")
                st.pyplot()

        if st.checkbox("Bar Plot for Prognosis"):
            plt.figure(figsize=(10, 5))
            plt.xticks(rotation=90)
            sns.barplot(df.prognosis.value_counts().index, df.prognosis.value_counts());
            plt.xlabel('prognosis', fontsize=15)
            plt.ylabel('count', fontsize=15)
            st.pyplot()

        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        def get_table_download_link(df):
            val = to_excel(df)
            b64 = base64.b64encode(val)  # val looks like b'...'
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="csv_disease.xlsx">Download Disease CSV file</a>'

        if st.button("Download link for Disease Prediction CSV File"):
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)



    if choice == "Predictor Tool":
        st.subheader("Predictor Tool")

        def get_value(val, my_dict):
            for key, value in my_dict.items():
                if val == key:
                    return value

        HTML_BANNER = """
            <div style="background-color:#B8FF48;padding:10px;border-radius:10px">
            <h1 style="color:black;text-align:center;">Choose the symptoms</h1>
            </div>
            """
        stc.html(HTML_BANNER)

        # Prediction
        st.subheader("Select the options to predict")

        itch = {"No": 0, "Yes": 1}
        choice_itch = st.radio("Itching?", tuple(itch.keys()))
        result_itch = get_value(choice_itch, itch)

        skin_rash = {"No": 0, "Yes": 1}
        choice_skin_rash = st.radio("Skin Rashes?", tuple(skin_rash.keys()))
        result_skin_rash = get_value(choice_skin_rash, skin_rash)

        nodal_skin_erup = {"No": 0, "Yes": 1}
        choice_nodalskin = st.radio("Nodal skin eruptions?", tuple(nodal_skin_erup.keys()))
        result_nodalskin = get_value(choice_nodalskin, nodal_skin_erup)

        con_snz = {"No": 0, "Yes": 1}
        choice_consnz = st.radio("Continuous Sneezing?", tuple(con_snz.keys()))
        result_consnz = get_value(choice_consnz, con_snz)

        shivering = {"No": 0, "Yes": 1}
        choice_shiv = st.radio("Shivering?", tuple(shivering.keys()))
        result_shiv = get_value(choice_shiv, shivering)

        chills = {"No": 0, "Yes": 1}
        choice_chills = st.radio("Chills?", tuple(chills.keys()))
        result_chills = get_value(choice_chills, chills)

        joint_pain = {"No": 0, "Yes": 1}
        choice_jp = st.radio("Joint Pain?", tuple(joint_pain.keys()))
        result_jp = get_value(choice_jp, joint_pain)

        stomach_pain = {"No": 0, "Yes": 1}
        choice_sp = st.radio("Stomach pain?", tuple(stomach_pain.keys()))
        result_sp = get_value(choice_sp, stomach_pain)

        acidity = {"No": 0, "Yes": 1}
        choice_acidity = st.radio("Acidity?", tuple(acidity.keys()))
        result_acidity = get_value(choice_acidity, acidity)

        ulc_ton = {"No": 0, "Yes": 1}
        choice_ulcton = st.radio("Ulcers on tongue?", tuple(ulc_ton.keys()))
        result_ulcton = get_value(choice_ulcton, ulc_ton)

        mus_wast = {"No": 0, "Yes": 1}
        choice_muswaste = st.radio("Muscle Wasting? (loss in muscle mass)", tuple(mus_wast.keys()))
        result_muswaste = get_value(choice_muswaste, mus_wast)

        vomiting = {"No": 0, "Yes": 1}
        choice_vom = st.radio("Vomiting?", tuple(vomiting.keys()))
        result_vom = get_value(choice_vom, vomiting)

        bur_mic = {"No": 0, "Yes": 1}
        choice_burmic = st.radio("Burning Micturition (Urination)?", tuple(bur_mic.keys()))
        result_burmic = get_value(choice_burmic, bur_mic)

        spot_ur = {"No": 0, "Yes": 1}
        choice_spur = st.radio("Spotting urination? (discoloration)", tuple(spot_ur.keys()))
        result_spur = get_value(choice_spur, spot_ur)

        fatigue = {"No": 0, "Yes": 1}
        choice_fat = st.radio("Fatigue?", tuple(fatigue.keys()))
        result_fat = get_value(choice_fat, fatigue)

        wt_gain = {"No": 0, "Yes": 1}
        choice_wtgn = st.radio("Weight Gain?", tuple(wt_gain.keys()))
        result_wtgn = get_value(choice_wtgn, wt_gain)

        anx = {"No": 0, "Yes": 1}
        choice_anx = st.radio("Anxiety?", tuple(anx.keys()))
        result_anx = get_value(choice_anx, anx)

        chf = {"No": 0, "Yes": 1}
        choice_chf = st.radio("Cold hands and feet?", tuple(chf.keys()))
        result_chf = get_value(choice_chf, chf)

        mood_swings = {"No": 0, "Yes": 1}
        choice_mood_swings = st.radio("Mood Swings?", tuple(mood_swings.keys()))
        result_mood_swings = get_value(choice_mood_swings, mood_swings)

        wt_loss = {"No": 0, "Yes": 1}
        choice_loss = st.radio("Weight loss?", tuple(wt_loss.keys()))
        result_loss = get_value(choice_loss, wt_loss)

        rstlsns = {"No": 0, "Yes": 1}
        choice_rstlsns = st.radio("Restlessness?", tuple(rstlsns.keys()))
        result_rstlsns = get_value(choice_rstlsns, rstlsns)

        lethargy = {"No": 0, "Yes": 1}
        choice_lethargy = st.radio("Lethargy?", tuple(lethargy.keys()))
        result_lethargy = get_value(choice_lethargy, lethargy)

        throat_patch = {"No": 0, "Yes": 1}
        choice_thtpat = st.radio("Patches in the Throat?", tuple(throat_patch.keys()))
        result_thtpat = get_value(choice_thtpat, throat_patch)

        irr_sr_lv = {"No": 0, "Yes": 1}
        choice_irrsrlv = st.radio("Irregular Sugar Level?", tuple(irr_sr_lv.keys()))
        result_irrsrlv = get_value(choice_irrsrlv, irr_sr_lv)

        cough = {"No": 0, "Yes": 1}
        choice_cough = st.radio("Cough?", tuple(cough.keys()))
        result_cough = get_value(choice_cough, cough)

        high_fever = {"No": 0, "Yes": 1}
        choice_hf = st.radio("High Fever?", tuple(high_fever.keys()))
        result_hf = get_value(choice_lethargy, high_fever)

        sunk_eyes = {"No": 0, "Yes": 1}
        choice_seye = st.radio("Sunk Eyes?", tuple(sunk_eyes.keys()))
        result_seye = get_value(choice_seye, sunk_eyes)

        breathlessness = {"No": 0, "Yes": 1}
        choice_btlsns = st.radio("Breathlessness?", tuple(breathlessness.keys()))
        result_btlsns = get_value(choice_btlsns, breathlessness)

        sweating = {"No": 0, "Yes": 1}
        choice_sweating = st.radio("Sweating?", tuple(sweating.keys()))
        result_sweating = get_value(choice_sweating, sweating)

        dehydration = {"No": 0, "Yes": 1}
        choice_dhd = st.radio("Dehydration?", tuple(dehydration.keys()))
        result_dhd = get_value(choice_dhd, dehydration)

        indigestion = {"No": 0, "Yes": 1}
        choice_indigestion = st.radio("Indigestion?", tuple(indigestion.keys()))
        result_indigestion = get_value(choice_indigestion, indigestion)

        headache = {"No": 0, "Yes": 1}
        choice_headache = st.radio("Headache?", tuple(headache.keys()))
        result_headache = get_value(choice_headache, headache)

        yellow_skin = {"No": 0, "Yes": 1}
        choice_ylwskin = st.radio("Yellow Skin?", tuple(yellow_skin.keys()))
        result_ylwskin = get_value(choice_ylwskin, yellow_skin)

        dark_urine = {"No": 0, "Yes": 1}
        choice_drkur = st.radio("Dark Urine?", tuple(dark_urine.keys()))
        result_drkur = get_value(choice_drkur, dark_urine)

        nausea = {"No": 0, "Yes": 1}
        choice_nausea = st.radio("Nausea?", tuple(nausea.keys()))
        result_nausea = get_value(choice_nausea, nausea)

        loss_appt = {"No": 0, "Yes": 1}
        choice_apploss = st.radio("Loss of appetite?", tuple(loss_appt.keys()))
        result_apploss = get_value(choice_apploss, loss_appt)

        pain_beh_eye = {"No": 0, "Yes": 1}
        choice_beheye = st.radio("Pain behind eyes?", tuple(pain_beh_eye.keys()))
        result_beheye = get_value(choice_beheye, pain_beh_eye)

        back_pain = {"No": 0, "Yes": 1}
        choice_bp = st.radio("Back Pain?", tuple(back_pain.keys()))
        result_bp = get_value(choice_bp, back_pain)

        constipation = {"No": 0, "Yes": 1}
        choice_csptn = st.radio("Constipation?", tuple(constipation.keys()))
        result_csptn = get_value(choice_csptn, constipation)

        abd_pain = {"No": 0, "Yes": 1}
        choice_abdpain = st.radio("Abdominal Pain?", tuple(abd_pain.keys()))
        result_abdpain = get_value(choice_abdpain, abd_pain)

        diar = {"No": 0, "Yes": 1}
        choice_diar = st.radio("Diarrhoea?", tuple(diar.keys()))
        result_diar = get_value(choice_diar, diar)

        mild_fever = {"No": 0, "Yes": 1}
        choice_mf = st.radio("Mild Fever?", tuple(mild_fever.keys()))
        result_mf = get_value(choice_mf, mild_fever)

        ylw_urine = {"No": 0, "Yes": 1}
        choice_ylwurine = st.radio("Yellow Urine?", tuple(ylw_urine.keys()))
        result_ylwurine = get_value(choice_ylwurine, ylw_urine)

        ylw_eyes = {"No": 0, "Yes": 1}
        choice_ylweyes = st.radio("Yellow Eyes?", tuple(ylw_eyes.keys()))
        result_ylweyes = get_value(choice_ylweyes, ylw_eyes)

        acute_liver_failure = {"No": 0, "Yes": 1}
        choice_alf = st.radio("Acute Liver Failure?", tuple(acute_liver_failure.keys()))
        result_alf = get_value(choice_alf, acute_liver_failure)

        fluid_ovld = {"No": 0, "Yes": 1}
        choice_fluidovd = st.radio("Overload of pus fluid?", tuple(fluid_ovld.keys()))
        result_fluidovd = get_value(choice_fluidovd, fluid_ovld)

        swollen_stomach = {"No": 0, "Yes": 1}
        choice_swl_sto = st.radio("Swelling of the stomach?", tuple(swollen_stomach.keys()))
        result_swl_sto = get_value(choice_swl_sto, swollen_stomach)

        swollen_lymph_nodes = {"No": 0, "Yes": 1}
        choice_sw_ly_nd = st.radio("Swollen Lymph Nodes?", tuple(swollen_lymph_nodes.keys()))
        result_sw_ly_nd = get_value(choice_sw_ly_nd, swollen_lymph_nodes)

        malaise = {"No": 0, "Yes": 1}
        choice_malaise = st.radio("Malaise (uneasiness/discomfort)?", tuple(malaise.keys()))
        result_malaise = get_value(choice_malaise, malaise)

        blurry_vision = {"No": 0, "Yes": 1}
        choice_blurry_vision = st.radio("Blurry or Distorted vision?", tuple(blurry_vision.keys()))
        result_blurry_vision = get_value(choice_blurry_vision, blurry_vision)

        phlegm = {"No": 0, "Yes": 1}
        choice_phlegm = st.radio("Phlegm?", tuple(phlegm.keys()))
        result_phlegm = get_value(choice_phlegm, phlegm)

        throat_irritation = {"No": 0, "Yes": 1}
        choice_throat_irritation = st.radio("Throat irritation?", tuple(throat_irritation.keys()))
        result_throat_irritation = get_value(choice_throat_irritation, throat_irritation)

        redness_eyes = {"No": 0, "Yes": 1}
        choice_red_eyes = st.radio("Red eyes?", tuple(redness_eyes.keys()))
        result_red_eyes = get_value(choice_red_eyes, redness_eyes)

        sinus_pressure = {"No": 0, "Yes": 1}
        choice_sinus_pressure = st.radio("Sinus Pressure?", tuple(sinus_pressure.keys()))
        result_sinus_pressure = get_value(choice_sinus_pressure, sinus_pressure)

        runny_nose = {"No": 0, "Yes": 1}
        choice_runny_nose = st.radio("Runny Nose?", tuple(runny_nose.keys()))
        result_runny_nose = get_value(choice_runny_nose, runny_nose)

        congestion = {"No": 0, "Yes": 1}
        choice_congestion = st.radio("Congestion?", tuple(congestion.keys()))
        result_congestion = get_value(choice_congestion, congestion)

        chest_pain = {"No": 0, "Yes": 1}
        choice_chest_pain = st.radio("Chest Pain?", tuple(chest_pain.keys()))
        result_chest_pain = get_value(choice_chest_pain, chest_pain)

        limb_weak = {"No": 0, "Yes": 1}
        choice_limb_weak = st.radio("Weakness in Limbs?", tuple(limb_weak.keys()))
        result_limb_weak = get_value(choice_limb_weak, limb_weak)

        fast_heart_rate = {"No": 0, "Yes": 1}
        choice_fast_heart_rate = st.radio("Fast Heart Rate?", tuple(fast_heart_rate.keys()))
        result_fast_heart_rate = get_value(choice_fast_heart_rate, fast_heart_rate)

        pain_bowel = {"No": 0, "Yes": 1}
        choice_pain_bowel = st.radio("Pain during bowel movements?", tuple(pain_bowel.keys()))
        result_pain_bowel = get_value(choice_pain_bowel, pain_bowel)

        pain_anal = {"No": 0, "Yes": 1}
        choice_pain_anal = st.radio("Pain in Anal Region?", tuple(pain_anal.keys()))
        result_pain_anal = get_value(choice_pain_anal, pain_anal)

        bloody_stool = {"No": 0, "Yes": 1}
        choice_bloody_stool = st.radio("Bloody Stool?", tuple(bloody_stool.keys()))
        result_bloody_stool = get_value(choice_bloody_stool, bloody_stool)

        anus_irritation = {"No": 0, "Yes": 1}
        choice_anus_irritation = st.radio("Irritation in the anus?", tuple(anus_irritation.keys()))
        result_anus_irritation = get_value(choice_anus_irritation, anus_irritation)

        neck_pain = {"No": 0, "Yes": 1}
        choice_neck_pain = st.radio("Neck Pain?", tuple(neck_pain.keys()))
        result_neck_pain = get_value(choice_neck_pain, neck_pain)

        dizziness = {"No": 0, "Yes": 1}
        choice_dizziness = st.radio("Dizziness?", tuple(dizziness.keys()))
        result_dizziness = get_value(choice_dizziness, dizziness)

        cramps = {"No": 0, "Yes": 1}
        choice_cramps = st.radio("Cramps?", tuple(cramps.keys()))
        result_cramps = get_value(choice_cramps, cramps)

        bruising = {"No": 0, "Yes": 1}
        choice_bruising = st.radio("Bruising?", tuple(bruising.keys()))
        result_bruising = get_value(choice_bruising, bruising)

        obesity = {"No": 0, "Yes": 1}
        choice_obesity = st.radio("Obesity?", tuple(obesity.keys()))
        result_obesity = get_value(choice_obesity, obesity)

        swollen_legs = {"No": 0, "Yes": 1}
        choice_swollen_legs = st.radio("Swollen legs?", tuple(swollen_legs.keys()))
        result_swollen_legs = get_value(choice_swollen_legs, swollen_legs)

        swollen_blood_vessels = {"No": 0, "Yes": 1}
        choice_swollen_blood_vessels = st.radio("Swollen Blood Vessels?", tuple(swollen_blood_vessels.keys()))
        result_swollen_blood_vessels = get_value(choice_swollen_blood_vessels, swollen_blood_vessels)

        puffy_face = {"No": 0, "Yes": 1}
        choice_puffy_face = st.radio("Puffy Face?", tuple(puffy_face.keys()))
        result_puffy_face = get_value(choice_puffy_face, puffy_face)

        enlarged_thyroid = {"No": 0, "Yes": 1}
        choice_enlarged_thyroid = st.radio("Enlarged Thyroid?", tuple(enlarged_thyroid.keys()))
        result_enlarged_thyroid = get_value(choice_enlarged_thyroid, enlarged_thyroid)

        brittle_nails = {"No": 0, "Yes": 1}
        choice_brittle_nails = st.radio("Brittle Nails?", tuple(brittle_nails.keys()))
        result_brittle_nails = get_value(choice_brittle_nails, brittle_nails)

        swollen_extremities = {"No": 0, "Yes": 1}
        choice_swollen_extremities = st.radio("Swollen Extremities?", tuple(swollen_extremities.keys()))
        result_swollen_extremities = get_value(choice_swollen_extremities, swollen_extremities)

        excessive_hunger = {"No": 0, "Yes": 1}
        choice_excessive_hunger = st.radio("Excessive Hunger?", tuple(excessive_hunger.keys()))
        result_excessive_hunger = get_value(choice_excessive_hunger, excessive_hunger)

        extra_marital_contacts = {"No": 0, "Yes": 1}
        choice_extra_marital_contacts = st.radio("Extra Marital Contacts?", tuple(extra_marital_contacts.keys()))
        result_extra_marital_contacts = get_value(choice_extra_marital_contacts, extra_marital_contacts)

        drying_and_tingling_lips = {"No": 0, "Yes": 1}
        choice_drying_and_tingling_lips = st.radio("Drying and tingling lips?", tuple(drying_and_tingling_lips.keys()))
        result_drying_and_tingling_lips = get_value(choice_drying_and_tingling_lips, drying_and_tingling_lips)

        slurred_speech = {"No": 0, "Yes": 1}
        choice_slurred_speech = st.radio("Slurred Speech?", tuple(slurred_speech.keys()))
        result_slurred_speech = get_value(choice_slurred_speech, slurred_speech)

        knee_pain = {"No": 0, "Yes": 1}
        choice_knee_pain = st.radio("Knee Pain?", tuple(knee_pain.keys()))
        result_knee_pain = get_value(choice_knee_pain, knee_pain)

        hip_joint_pain = {"No": 0, "Yes": 1}
        choice_hip_joint_pain = st.radio("Hip joint pain?", tuple(hip_joint_pain.keys()))
        result_hip_joint_pain = get_value(choice_hip_joint_pain, hip_joint_pain)

        muscle_weakness = {"No": 0, "Yes": 1}
        choice_muscle_weakness = st.radio("Muscle Weakness?", tuple(muscle_weakness.keys()))
        result_muscle_weakness = get_value(choice_muscle_weakness, muscle_weakness)

        stiff_neck = {"No": 0, "Yes": 1}
        choice_stiff_neck = st.radio("Stiff Neck?", tuple(stiff_neck.keys()))
        result_stiff_neck = get_value(choice_stiff_neck, stiff_neck)

        swelling_joints = {"No": 0, "Yes": 1}
        choice_swelling_joints = st.radio("Swelling Joints?", tuple(swelling_joints.keys()))
        result_swelling_joints = get_value(choice_swelling_joints, swelling_joints)

        movement_stiffness = {"No": 0, "Yes": 1}
        choice_movement_stiffness = st.radio("Movement Stiffness?", tuple(movement_stiffness.keys()))
        result_movement_stiffness = get_value(choice_movement_stiffness, movement_stiffness)

        spinning_movements = {"No": 0, "Yes": 1}
        choice_spinning_movements = st.radio("Spinning Movements?", tuple(spinning_movements.keys()))
        result_spinning_movements = get_value(choice_spinning_movements, spinning_movements)

        loss_of_balance = {"No": 0, "Yes": 1}
        choice_loss_of_balance = st.radio("Loss of Balance?", tuple(loss_of_balance.keys()))
        result_loss_of_balance = get_value(choice_loss_of_balance, loss_of_balance)

        unsteadiness = {"No": 0, "Yes": 1}
        choice_unsteadiness = st.radio("Unsteadiness?", tuple(unsteadiness.keys()))
        result_unsteadiness = get_value(choice_unsteadiness, unsteadiness)

        weakness_of_one_body_side = {"No": 0, "Yes": 1}
        choice_weakness_of_one_body_side = st.radio("Weakness in one side of the body?",
                                                    tuple(weakness_of_one_body_side.keys()))
        result_weakness_of_one_body_side = get_value(choice_weakness_of_one_body_side, weakness_of_one_body_side)

        loss_of_smell = {"No": 0, "Yes": 1}
        choice_loss_of_smell = st.radio("Loss of smell?", tuple(loss_of_smell.keys()))
        result_loss_of_smell = get_value(choice_loss_of_smell, loss_of_smell)

        bladder_discomfort = {"No": 0, "Yes": 1}
        choice_bladder_discomfort = st.radio("Bladder discomfort?", tuple(bladder_discomfort.keys()))
        result_bladder_discomfort = get_value(choice_bladder_discomfort, bladder_discomfort)

        foul_smell_urine = {"No": 0, "Yes": 1}
        choice_foul_smell_urine = st.radio("Foul Smell of Urine?", tuple(foul_smell_urine.keys()))
        result_foul_smell_urine = get_value(choice_foul_smell_urine, foul_smell_urine)

        continuous_feel_of_urine = {"No": 0, "Yes": 1}
        choice_continuous_feel_of_urine = st.radio("Continuous Feel of Urine?", tuple(continuous_feel_of_urine.keys()))
        result_continuous_feel_of_urine = get_value(choice_continuous_feel_of_urine, continuous_feel_of_urine)

        passage_of_gases = {"No": 0, "Yes": 1}
        choice_passage_of_gases = st.radio("Passage of gases?", tuple(passage_of_gases.keys()))
        result_passage_of_gases = get_value(choice_passage_of_gases, passage_of_gases)

        internal_itching = {"No": 0, "Yes": 1}
        choice_internal_itching = st.radio("Internal Itching?", tuple(internal_itching.keys()))
        result_internal_itching = get_value(choice_internal_itching, internal_itching)

        toxic_look = {"No": 0, "Yes": 1}
        choice_toxic_look = st.radio("Toxic Look (Typhos)?", tuple(toxic_look.keys()))
        result_toxic_look = get_value(choice_toxic_look, toxic_look)

        depression = {"No": 0, "Yes": 1}
        choice_depression = st.radio("Suffering from depression?", tuple(depression.keys()))
        result_depression = get_value(choice_depression, depression)

        irritability = {"No": 0, "Yes": 1}
        choice_irritability = st.radio("Irritability?", tuple(irritability.keys()))
        result_irritability = get_value(choice_irritability, irritability)

        muscle_pain = {"No": 0, "Yes": 1}
        choice_muscle_pain = st.radio("Muscle Pain?", tuple(muscle_pain.keys()))
        result_muscle_pain = get_value(choice_muscle_pain, muscle_pain)

        altered_sensorium = {"No": 0, "Yes": 1}
        choice_altered_sensorium = st.radio("Altered Sensorium?", tuple(altered_sensorium.keys()))
        result_altered_sensorium = get_value(choice_altered_sensorium, altered_sensorium)

        red_spots_over_body = {"No": 0, "Yes": 1}
        choice_red_spots_over_body = st.radio("Red spots over Body?", tuple(red_spots_over_body.keys()))
        result_red_spots_over_body = get_value(choice_red_spots_over_body, red_spots_over_body)

        belly_pain = {"No": 0, "Yes": 1}
        choice_belly_pain = st.radio("Belly Pain?", tuple(belly_pain.keys()))
        result_belly_pain = get_value(choice_belly_pain, belly_pain)

        abnormal_menstruation = {"No": 0, "Yes": 1}
        choice_abnormal_menstruation = st.radio("Abnormal Menstruation?", tuple(abnormal_menstruation.keys()))
        result_abnormal_menstruation = get_value(choice_abnormal_menstruation, abnormal_menstruation)

        dyschromic_patches = {"No": 0, "Yes": 1}
        choice_dyschromic_patches = st.radio("Dyschromic Patches?", tuple(dyschromic_patches.keys()))
        result_dyschromic_patches = get_value(choice_dyschromic_patches, dyschromic_patches)

        watering_from_eyes = {"No": 0, "Yes": 1}
        choice_watering_from_eyes = st.radio("Watering from eyes?", tuple(watering_from_eyes.keys()))
        result_watering_from_eyes = get_value(choice_watering_from_eyes, watering_from_eyes)

        increased_appetite = {"No": 0, "Yes": 1}
        choice_increased_appetite = st.radio("Increased appetite?", tuple(increased_appetite.keys()))
        result_increased_appetite = get_value(choice_increased_appetite, increased_appetite)

        polyuria = {"No": 0, "Yes": 1}
        choice_polyuria = st.radio("Polyuria (Excessive peeing)?", tuple(polyuria.keys()))
        result_polyuria = get_value(choice_polyuria, polyuria)

        family_history = {"No": 0, "Yes": 1}
        choice_family_history = st.radio("Family History?", tuple(family_history.keys()))
        result_family_history = get_value(choice_family_history, family_history)

        mucoid_sputum = {"No": 0, "Yes": 1}
        choice_mucoid_sputum = st.radio("Presence of Mucoid Sputum?", tuple(mucoid_sputum.keys()))
        result_mucoid_sputum = get_value(choice_mucoid_sputum, mucoid_sputum)

        rusty_sputum = {"No": 0, "Yes": 1}
        choice_rusty_sputum = st.radio("Presence of Rusty Sputum?", tuple(rusty_sputum.keys()))
        result_rusty_sputum = get_value(choice_rusty_sputum, rusty_sputum)

        lack_of_concentration = {"No": 0, "Yes": 1}
        choice_lack_of_concentration = st.radio("Lack of Concentration?", tuple(lack_of_concentration.keys()))
        result_lack_of_concentration = get_value(choice_lack_of_concentration, lack_of_concentration)

        visual_disturbances = {"No": 0, "Yes": 1}
        choice_visual_disturbances = st.radio("Visual Disturbances?", tuple(visual_disturbances.keys()))
        result_visual_disturbances = get_value(choice_visual_disturbances, visual_disturbances)

        receiving_blood_transfusion = {"No": 0, "Yes": 1}
        choice_receiving_blood_transfusion = st.radio("Receiving Blood Transfusion?",
                                                      tuple(receiving_blood_transfusion.keys()))
        result_receiving_blood_transfusion = get_value(choice_receiving_blood_transfusion, receiving_blood_transfusion)

        receiving_unsterile_injections = {"No": 0, "Yes": 1}
        choice_receiving_unsterile_injections = st.radio("Receiving Unsterile Injections?",
                                                         tuple(receiving_unsterile_injections.keys()))
        result_receiving_unsterile_injections = get_value(choice_receiving_unsterile_injections,
                                                          receiving_unsterile_injections)

        coma = {"No": 0, "Yes": 1}
        choice_coma = st.radio("Coma?", tuple(coma.keys()))
        result_coma = get_value(choice_coma, coma)

        stomach_bleeding = {"No": 0, "Yes": 1}
        choice_stomach_bleeding = st.radio("Stomach Bleeding?", tuple(stomach_bleeding.keys()))
        result_stomach_bleeding = get_value(choice_stomach_bleeding, stomach_bleeding)

        distention_of_abdomen = {"No": 0, "Yes": 1}
        choice_distention_of_abdomen = st.radio("Distention of Abdomen?", tuple(distention_of_abdomen.keys()))
        result_distention_of_abdomen = get_value(choice_distention_of_abdomen, distention_of_abdomen)

        history_of_alcohol_consumption = {"No": 0, "Yes": 1}
        choice_history_of_alcohol_consumption = st.radio("History of Alcohol Consumption?",
                                                         tuple(history_of_alcohol_consumption.keys()))
        result_history_of_alcohol_consumption = get_value(choice_history_of_alcohol_consumption,
                                                          history_of_alcohol_consumption)

        fluid_overload = {"No": 0, "Yes": 1}
        choice_fluid_overload = st.radio("Fluid Overload?", tuple(fluid_overload.keys()))
        result_fluid_overload = get_value(choice_fluid_overload, fluid_overload)

        blood_in_sputum = {"No": 0, "Yes": 1}
        choice_blood_in_sputum = st.radio("Blood in Sputum?", tuple(blood_in_sputum.keys()))
        result_blood_in_sputum = get_value(choice_blood_in_sputum, blood_in_sputum)

        prominent_veins_on_calf = {"No": 0, "Yes": 1}
        choice_prominent_veins_on_calf = st.radio("Prominent Veins on Calf?", tuple(prominent_veins_on_calf.keys()))
        result_prominent_veins_on_calf = get_value(choice_prominent_veins_on_calf, prominent_veins_on_calf)

        palpitations = {"No": 0, "Yes": 1}
        choice_palpitations = st.radio("Palpitations?", tuple(palpitations.keys()))
        result_palpitations = get_value(choice_palpitations, palpitations)

        painful_walking = {"No": 0, "Yes": 1}
        choice_painful_walking = st.radio("Painful Walking?", tuple(painful_walking.keys()))
        result_painful_walking = get_value(choice_painful_walking, painful_walking)

        pus_filled_pimples = {"No": 0, "Yes": 1}
        choice_pus_filled_pimples = st.radio("Pus-Filled Pimples?", tuple(pus_filled_pimples.keys()))
        result_pus_filled_pimples = get_value(choice_pus_filled_pimples, pus_filled_pimples)

        blackheads = {"No": 0, "Yes": 1}
        choice_blackheads = st.radio("Blackheads?", tuple(blackheads.keys()))
        result_blackheads = get_value(choice_blackheads, blackheads)

        scarring = {"No": 0, "Yes": 1}
        choice_scarring = st.radio("Scarring?", tuple(scarring.keys()))
        result_scarring = get_value(choice_scarring, scarring)

        skin_peeling = {"No": 0, "Yes": 1}
        choice_skin_peeling = st.radio("Skin Peeling?", tuple(skin_peeling.keys()))
        result_skin_peeling = get_value(choice_skin_peeling, skin_peeling)

        silver_like_dusting_scaly = {"No": 0, "Yes": 1}
        choice_silver_like_dusting_scaly = st.radio("Scaly skin?", tuple(silver_like_dusting_scaly.keys()))
        result_silver_like_dusting_scaly = get_value(choice_silver_like_dusting_scaly, silver_like_dusting_scaly)

        small_dents_in_nails = {"No": 0, "Yes": 1}
        choice_small_dents_in_nails = st.radio("Dents in nails?", tuple(small_dents_in_nails.keys()))
        result_small_dents_in_nails = get_value(choice_small_dents_in_nails, small_dents_in_nails)

        inflammatory_nails = {"No": 0, "Yes": 1}
        choice_inflammatory_nails = st.radio("Inflammatory Nails?", tuple(inflammatory_nails.keys()))
        result_inflammatory_nails = get_value(choice_inflammatory_nails, inflammatory_nails)

        blister = {"No": 0, "Yes": 1}
        choice_blister = st.radio("Blisters?", tuple(blister.keys()))
        result_blister = get_value(choice_blister, blister)

        red_sore_around_nose = {"No": 0, "Yes": 1}
        choice_red_sore_around_nose = st.radio("Red Soreness Around the Nose?", tuple(red_sore_around_nose.keys()))
        result_red_sore_around_nose = get_value(choice_red_sore_around_nose, red_sore_around_nose)

        yellow_crust_ooze = {"No": 0, "Yes": 1}
        choice_yellow_crust_ooze = st.radio("Yellow Scab/Crust Ooze?", tuple(yellow_crust_ooze.keys()))
        result_yellow_crust_ooze = get_value(choice_yellow_crust_ooze, yellow_crust_ooze)

        # Result and in json format
        results = [result_itch,
                   result_skin_rash,
                   result_nodalskin,
                   result_consnz,
                   result_shiv,
                   result_chills,
                   result_jp,
                   result_sp,
                   result_acidity,
                   result_ulcton,
                   result_muswaste,
                   result_vom,
                   result_burmic,
                   result_spur,
                   result_fat,
                   result_wtgn,
                   result_anx,
                   result_chf,
                   result_mood_swings,
                   result_loss,
                   result_rstlsns,
                   result_lethargy,
                   result_thtpat,
                   result_irrsrlv,
                   result_cough,
                   result_hf,
                   result_seye,
                   result_btlsns,
                   result_sweating,
                   result_dhd,
                   result_indigestion,
                   result_headache,
                   result_ylwskin,
                   result_drkur,
                   result_nausea,
                   result_apploss,
                   result_beheye,
                   result_bp,
                   result_csptn,
                   result_abdpain,
                   result_diar,
                   result_mf,
                   result_ylwurine,
                   result_ylweyes,
                   result_alf,
                   result_fluidovd,
                   result_swl_sto,
                   result_sw_ly_nd,
                   result_malaise,
                   result_blurry_vision,
                   result_phlegm,
                   result_throat_irritation,
                   result_red_eyes,
                   result_sinus_pressure,
                   result_runny_nose,
                   result_congestion,
                   result_chest_pain,
                   result_limb_weak,
                   result_fast_heart_rate,
                   result_pain_bowel,
                   result_pain_anal,
                   result_bloody_stool,
                   result_anus_irritation,
                   result_neck_pain,
                   result_dizziness,
                   result_cramps,
                   result_bruising,
                   result_obesity,
                   result_swollen_legs,
                   result_swollen_blood_vessels,
                   result_puffy_face,
                   result_enlarged_thyroid,
                   result_brittle_nails,
                   result_swollen_extremities,
                   result_excessive_hunger,
                   result_extra_marital_contacts,
                   result_drying_and_tingling_lips,
                   result_slurred_speech,
                   result_knee_pain,
                   result_hip_joint_pain,
                   result_muscle_weakness,
                   result_stiff_neck,
                   result_swelling_joints,
                   result_movement_stiffness,
                   result_spinning_movements,
                   result_loss_of_balance,
                   result_unsteadiness,
                   result_weakness_of_one_body_side,
                   result_loss_of_smell,
                   result_bladder_discomfort,
                   result_foul_smell_urine,
                   result_continuous_feel_of_urine,
                   result_passage_of_gases,
                   result_internal_itching,
                   result_toxic_look,
                   result_depression,
                   result_irritability,
                   result_muscle_pain,
                   result_altered_sensorium,
                   result_red_spots_over_body,
                   result_belly_pain,
                   result_abnormal_menstruation,
                   result_dyschromic_patches,
                   result_watering_from_eyes,
                   result_increased_appetite,
                   result_polyuria,
                   result_family_history,
                   result_mucoid_sputum,
                   result_rusty_sputum,
                   result_lack_of_concentration,
                   result_visual_disturbances,
                   result_receiving_blood_transfusion,
                   result_receiving_unsterile_injections,
                   result_coma,
                   result_stomach_bleeding,
                   result_history_of_alcohol_consumption,
                   result_fluid_overload,
                   result_blood_in_sputum,
                   result_prominent_veins_on_calf,
                   result_palpitations,
                   result_painful_walking,
                   result_pus_filled_pimples,
                   result_blackheads,
                   result_scarring,
                   result_skin_peeling,
                   result_silver_like_dusting_scaly,
                   result_small_dents_in_nails,
                   result_inflammatory_nails,
                   result_blister,
                   result_red_sore_around_nose,
                   result_yellow_crust_ooze
                   ]
        displayed_results = [choice_itch, choice_skin_rash, choice_nodalskin, choice_consnz, choice_shiv, choice_chills,
                             choice_jp, choice_sp,
                             choice_acidity, choice_ulcton, choice_muswaste, choice_vom, choice_burmic, choice_spur,
                             choice_fat, choice_wtgn, choice_anx, choice_chf,
                             choice_mood_swings, choice_loss, choice_rstlsns, choice_lethargy, choice_thtpat,
                             choice_irrsrlv,
                             choice_cough, choice_hf, choice_seye, choice_btlsns, choice_sweating,
                             choice_dhd, choice_indigestion, choice_headache, choice_ylwskin, choice_drkur,
                             choice_nausea,
                             choice_apploss, choice_beheye, choice_bp, choice_csptn, choice_abdpain, choice_diar,
                             choice_mf, choice_ylwurine, choice_ylweyes, choice_alf, choice_fluidovd, choice_swl_sto,
                             choice_sw_ly_nd, choice_malaise, choice_blurry_vision, choice_phlegm,
                             choice_throat_irritation,
                             choice_red_eyes,
                             choice_sinus_pressure, choice_runny_nose, choice_congestion, choice_chest_pain,
                             choice_limb_weak,
                             choice_fast_heart_rate, choice_pain_bowel, choice_pain_anal, choice_bloody_stool,
                             choice_anus_irritation,
                             choice_neck_pain, choice_dizziness, choice_cramps, choice_bruising, choice_obesity,
                             choice_swollen_legs, choice_swollen_blood_vessels,
                             choice_puffy_face, choice_enlarged_thyroid, choice_brittle_nails,
                             choice_swollen_extremities,
                             choice_excessive_hunger,
                             choice_extra_marital_contacts, choice_drying_and_tingling_lips, choice_slurred_speech,
                             choice_knee_pain, choice_hip_joint_pain, choice_muscle_weakness, choice_stiff_neck,
                             choice_swelling_joints,
                             choice_movement_stiffness, choice_spinning_movements, choice_loss_of_balance,
                             choice_unsteadiness,
                             choice_weakness_of_one_body_side, choice_loss_of_smell, choice_bladder_discomfort,
                             choice_foul_smell_urine,
                             choice_continuous_feel_of_urine, choice_passage_of_gases, choice_internal_itching,
                             choice_toxic_look,
                             choice_depression, choice_irritability, choice_muscle_pain, choice_altered_sensorium,
                             choice_red_spots_over_body,
                             choice_belly_pain, choice_abnormal_menstruation, choice_dyschromic_patches,
                             choice_watering_from_eyes,
                             choice_increased_appetite, choice_polyuria, choice_family_history, choice_mucoid_sputum,
                             choice_rusty_sputum, choice_lack_of_concentration, choice_visual_disturbances,
                             choice_receiving_blood_transfusion,
                             choice_receiving_unsterile_injections, choice_coma, choice_stomach_bleeding,
                             choice_history_of_alcohol_consumption,
                             choice_fluid_overload, choice_blood_in_sputum, choice_prominent_veins_on_calf,
                             choice_palpitations,
                             choice_painful_walking, choice_pus_filled_pimples, choice_blackheads, choice_scarring,
                             choice_skin_peeling,
                             choice_silver_like_dusting_scaly, choice_small_dents_in_nails, choice_inflammatory_nails,
                             choice_blister,
                             choice_red_sore_around_nose, choice_yellow_crust_ooze
                             ]

        prettified_result = {
            "itch": choice_itch,
            "skin rash": choice_skin_rash,
            "nodal_skin_erup": choice_nodalskin,
            "continuous_sneezing": choice_consnz,
            "shivering": choice_shiv,
            "chills": choice_chills,
            "joint_pain": choice_jp,
            "stomach_pain": choice_sp,
            "acidity": choice_acidity,
            "ulc_ton": choice_ulcton,
            "mus_wast": choice_muswaste,
            "vomiting": choice_vom,
            "bur_mic": choice_burmic,
            "spot_ur": choice_spur,
            "fatigue": choice_fat,
            "wt_gain": choice_wtgn,
            "anx": choice_anx,
            "chf": choice_chf,
            "mood_swings": choice_mood_swings,
            "wt_loss": choice_loss,
            "rstlsns": choice_rstlsns,
            "lethargy": choice_lethargy,
            "throat_patch": choice_thtpat,
            "irr_sr_lv": choice_irrsrlv,
            "cough": choice_cough,
            "high_fever": choice_hf,
            "sunk_eyes": choice_seye,
            "breathlessness": choice_btlsns,
            "sweating": choice_sweating,
            "dehydration": choice_dhd,
            "indigestion": choice_indigestion,
            "headache": choice_headache,
            "yellow_skin": choice_ylwskin,
            "dark_urine": choice_drkur,
            "nausea": choice_nausea,
            "loss_appt": choice_apploss,
            "pain_beh_eye": choice_beheye,
            "back_pain": choice_bp,
            "constipation": choice_csptn,
            "abd_pain": choice_abdpain,
            "diar": choice_diar,
            "mild_fever": choice_mf,
            "ylw_urine": choice_ylwurine,
            "ylw_eyes": choice_ylweyes,
            "acute_liver_failure": choice_alf,
            "fluid_ovld": choice_fluidovd,
            "swollen_stomach": choice_swl_sto,
            "swollen_lymph_nodes": choice_sw_ly_nd,
            "malaise": choice_malaise,
            "blurry_vision": choice_blurry_vision,
            "phlegm": choice_phlegm,
            "throat_irritation": choice_throat_irritation,
            "redness_eyes": choice_red_eyes,
            "sinus_pressure": choice_sinus_pressure,
            "runny_nose": choice_runny_nose,
            "congestion": choice_congestion,
            "chest_pain": choice_chest_pain,
            "limb_weak": choice_limb_weak,
            "fast_heart_rate": choice_fast_heart_rate,
            "pain_bowel": choice_pain_bowel,
            "pain_anal": choice_pain_anal,
            "bloody_stool": choice_bloody_stool,
            "anus_irritation": choice_anus_irritation,
            "neck_pain": choice_neck_pain,
            "dizziness": choice_dizziness,
            "cramps": choice_cramps,
            "bruising": choice_bruising,
            "obesity": choice_obesity,
            "swollen_legs": choice_swollen_legs,
            "swollen_blood_vessels": choice_swollen_blood_vessels,
            "puffy_face": choice_puffy_face,
            "enlarged_thyroid": choice_enlarged_thyroid,
            "brittle_nails": choice_brittle_nails,
            "swollen_extremities": choice_swollen_extremities,
            "excessive_hunger": choice_excessive_hunger,
            "extra_marital_contacts": choice_extra_marital_contacts,
            "drying_and_tingling_lips": choice_drying_and_tingling_lips,
            "slurred_speech": choice_slurred_speech,
            "knee_pain": choice_knee_pain,
            "hip_joint_pain": choice_hip_joint_pain,
            "muscle_weakness": choice_muscle_weakness,
            "stiff_neck": choice_stiff_neck,
            "swelling_joints": choice_swelling_joints,
            "movement_stiffness": choice_movement_stiffness,
            "spinning_movements": choice_spinning_movements,
            "loss_of_balance": choice_loss_of_balance,
            "unsteadiness": choice_unsteadiness,
            "weakness_of_one_body_side": choice_weakness_of_one_body_side,
            "loss_of_smell": choice_loss_of_smell,
            "bladder_discomfort": choice_bladder_discomfort,
            "foul_smell_urine": choice_foul_smell_urine,
            "continuous_feel_of_urine": choice_continuous_feel_of_urine,
            "passage_of_gases": choice_passage_of_gases,
            "internal_itching": choice_internal_itching,
            "toxic_look": choice_toxic_look,
            "depression": choice_depression,
            "irritability": choice_irritability,
            "muscle_pain": choice_muscle_pain,
            "altered_sensorium": choice_altered_sensorium,
            "red_spots_over_body": choice_red_spots_over_body,
            "belly_pain": choice_belly_pain,
            "abnormal_menstruation": choice_abnormal_menstruation,
            "dyschromic_patches": choice_dyschromic_patches,
            "watering_from_eyes": choice_watering_from_eyes,
            "increased_appetite": choice_increased_appetite,
            "polyuria": choice_polyuria,
            "family_history": choice_family_history,
            "mucoid_sputum": choice_mucoid_sputum,
            "rusty_sputum": choice_rusty_sputum,
            "lack_of_concentration": choice_lack_of_concentration,
            "visual_disturbances": choice_visual_disturbances,
            "receiving_blood_transfusion": choice_receiving_blood_transfusion,
            "receiving_unsterile_injections": choice_receiving_unsterile_injections,
            "coma": choice_coma,
            "stomach_bleeding": choice_stomach_bleeding,
            "history_of_alcohol_consumption": choice_history_of_alcohol_consumption,
            "fluid_overload": choice_fluid_overload,
            "blood_in_sputum": choice_blood_in_sputum,
            "prominent_veins_on_calf": choice_prominent_veins_on_calf,
            "palpitations": choice_palpitations,
            "painful_walking": choice_painful_walking,
            "pus_filled_pimples": choice_pus_filled_pimples,
            "blackheads": choice_blackheads,
            "scarring": choice_scarring,
            "skin_peeling": choice_skin_peeling,
            "silver_like_dusting_scaly": choice_silver_like_dusting_scaly,
            "small_dents_in_nails": choice_small_dents_in_nails,
            "inflammatory_nails": choice_inflammatory_nails,
            "blister": choice_blister,
            "red_sore_around_nose": choice_red_sore_around_nose,
            "yellow_crust_ooze": choice_yellow_crust_ooze
        }

        sample_data = np.array(results).reshape(1, -1)

        if st.checkbox("User Inputs Summary"):
            st.json(prettified_result)
            st.text("Vectorized as ::{}".format(results))

        st.subheader("Prediction Tool")
        if st.checkbox("Make Prediction"):
            all_ml_dict = {
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Neural Network-Multi-layer Perceptron Classifier': MLPClassifier()
            }

            # Find the Key From Dictionary
            def get_key(val, my_dict):
                for key, value in my_dict.items():
                    if val == value:
                        return key

            # Load Models
            def load_model_n_predict(model_file):
                loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
                return loaded_model

            # Model Selection
            model_choice = st.selectbox('Machine Learning Model Choice', list(all_ml_dict.keys()))
            if st.button("Predict"):
                if model_choice == 'Random Forest Classifier':
                    model_predictorrf = load_model_n_predict("rf_model_pickle.joblib")
                    prediction = model_predictorrf.predict(sample_data)
                # final_result = get_key(prediction,prediction_label)
                # st.info(final_result)
                elif model_choice == 'Decision Tree Classifier':
                    model_predictordtc = load_model_n_predict("dtc_model_pickle.joblib")
                    prediction = model_predictordtc.predict(sample_data)
                # st.text(prediction)
                elif model_choice == 'Neural Network-Multi-layer Perceptron Classifier':
                    model_predictornnmlp = load_model_n_predict("mlp_model_pickle.joblib")
                    prediction = model_predictornnmlp.predict(sample_data)

                st.markdown('**Your disease prediction according to the inputs**')
                st.subheader(prediction)

    if choice == "About":
        st.subheader("About")
        st.markdown('**Hosted on Google App Engine**')
        st.info("Medictionary Disease Prediction Tool")
        st.text("Made By Rakshit Saxena")

        st.text("Web Health Informatics, 2021")


if __name__ == '__main__':
    main()
