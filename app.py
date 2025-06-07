import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("model_predict_dropout.pkl")


model = load_model()


def categorical_input(label, options, help_text=""):
    return st.selectbox(label, options, help=help_text)


st.title("🎓Student Dropout Prediction")

st.write("Fill out this form to predict probability student dropout")

with st.form("student_form"):
    st.header("📋 Student Information")

    marital_status = categorical_input("Marital Status", ['Single', 'Married', 'Widower', 'Divorced', 'Facto union', 'Legally separated'],
                                       "The marital status of the student.")

    application_mode = categorical_input("Application Mode", ['1', '2', '5', '7', '10', '15', '16', '17', '18', '26', '27', '39', '43', '44', '51', '53', '57'],
                                         "Method of application used by the student. ")
    with st.expander("See List Code Application Mode"):
        st.markdown("""
        - **1** – 1st phase - general contingent
        - **2** – Ordinance No. 612/93
        - **5** – 1st phase - special contingent (Azores Island)
        - **7** – Holders of other higher courses
        - **10** – Ordinance No. 854-B/99
        - **15** – International student (bachelor)
        - **16** – 1st phase - special contingent (Madeira Island)
        - **17** – 2nd phase - general contingent
        - **18** – 3rd phase - general contingent
        - **26** – Ordinance No. 533-A/99, item b2) (Different Plan)
        - **27** – Ordinance No. 533-A/99, item b3 (Other Institution)
        - **39** – Over 23 years old
        - **43** – Change of course
        - **44** – Technological specialization diploma holders
        - **51** – Change of institution/course
        - **53** – Short cycle diploma holders
        - **57** – Change of institution/course (International)
        """)

    course = categorical_input("Code Course Taken By Student", [
                               '9556', '9119', '9773', '9238', '9500', '8014', '9991', '9670', '9003', '9130', '9085', '9254', '171', '9853', '9070', '9147', '33'],
                               "The course taken by the student."
                               )
    with st.expander("See List Code Course"):
        st.markdown("""
        - **33**- Biofuel Production Technologies
        - **171**- Animation and Multimedia Design
        - **8014** - Social Service (evening attendance) "
        - **9003** - Agronomy
        - **9070** - Communication Design
        - **9085** - Veterinary Nursing
        - **9119** - Informatics Engineering
        - **9130** - Equinculture
        - **9147** - Management
        - **9238** - Social Service
        - **9254** - Tourism
        - **9500** - Nursing
        - **9556** - Oral Hygiene
        - **9670** - Advertising and Marketing Management
        - **9773** - Journalism and Communication
        - **9853** - Basic Education
        - **9991** - Management (evening attendance)
        """)
    attendance = categorical_input("Time Attendance", ['Daytime', 'Evening'],
                                   "Whether the student attends classes during the day or in the evening. ")

    mothers_qualification = categorical_input("Mother Qualification", ['19', '37', '3', '34', '1', '38', '2', '29', '5', '12', '9', '4', '10', '30', '39', '40', '36', '26',
                                                                       '43', '11', '6', '41', '42', '27', '18', '14', '35', '22'], "The qualification of the student's mother"
                                              )

    fathers_qualification = categorical_input("Father Qualification", ['19', '37', '3', '34', '1', '38', '2', '29', '5', '12', '9', '4', '10', '30', '39', '40', '36', '26',
                                                                       '43', '11', '6', '41', '42', '27', '18', '14', '35', '22'], "The qualification of the student's father"
                                              )
    with st.expander("List Educational Qualification"):
        st.markdown("""
      - **1** – Secondary Education - 12th Year of Schooling or Equivalent  
      - **2** – Higher Education - Bachelor's Degree  
      - **3** – Higher Education - Degree  
      - **4** – Higher Education - Master's  
      - **5** – Higher Education - Doctorate  
      - **6** – Frequency of Higher Education  
      - **9** – 12th Year of Schooling - Not Completed  
      - **10** – 11th Year of Schooling - Not Completed  
      - **11** – 7th Year (Old)  
      - **12** – Other - 11th Year of Schooling  
      - **14** – 10th Year of Schooling  
      - **18** – General Commerce Course  
      - **19** – Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent  
      - **22** – Technical-Professional Course  
      - **26** – 7th Year of Schooling  
      - **27** – 2nd Cycle of the General High School Course  
      - **29** – 9th Year of Schooling - Not Completed  
      - **30** – 8th Year of Schooling  
      - **34** – Unknown  
      - **35** – Can't Read or Write  
      - **36** – Can Read Without Having a 4th Year of Schooling  
      - **37** – Basic Education 1st Cycle (4th/5th Year) or Equivalent  
      - **38** – Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent  
      - **39** – Technological Specialization Course  
      - **40** – Higher Education - Degree (1st Cycle)  
      - **41** – Specialized Higher Studies Course  
      - **42** – Professional Higher Technical Course  
      - **43** – Higher Education - Master (2nd Cycle)  
      - **44** – Higher Education - Doctorate (3rd Cycle)  
    """)
    mothers_occupation = categorical_input('Mother Occupation', ['7', '9', '5', '4', '0', '8', '3', '191', '90', '2', '6', '1', '144', '99', '194', '134', '123', '193',
                                                                 '141', '192', '152', '132', '175', '151', '10', '125', '131', '143', '153', '171', '173'], "The occupation of the student's mother."
                                           )
    fathers_occupation = categorical_input("Father Occupation", ['7', '6', '10', '9', '1', '0', '4', '5', '8', '171', '3', '182', '2', '90', '123', '154', '193', '195',
                                                                 '134', '103', '99', '144', '194', '102', '143', '192', '131', '163', '132', '112', '175',
                                                                 '174', '135', '151', '161', '124', '152', '101', '114', '121', '122', '141', '153', '172', '181', '183', '125'], "The occupation of the student's father."
                                           )
    with st.expander("List Occupation"):
        st.markdown("""
      - **0** – Student  
      - **1** – Representatives of Legislative/Executive Bodies, Directors, and Executive Managers  
      - **2** – Specialists in Intellectual and Scientific Activities  
      - **3** – Intermediate Level Technicians and Professions  
      - **4** – Administrative Staff  
      - **5** – Personal Services, Security and Safety Workers, and Sellers  
      - **6** – Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry  
      - **7** – Skilled Workers in Industry, Construction, and Craftsmen  
      - **8** – Installation and Machine Operators, Assembly Workers  
      - **9** – Unskilled Workers  
      - **10** – Armed Forces Professions  
      - **90** – Other Situation  
      - **99** – (blank)  
      - **122** – Health Professionals  
      - **123** – Teachers  
      - **125** – ICT Specialists  
      - **131** – Intermediate Level Science and Engineering Technicians  
      - **132** – Intermediate Level Health Technicians and Professionals  
      - **134** – Intermediate Level Technicians in Legal, Social, Sports, Cultural, and Similar Services  
      - **141** – Office Workers, Secretaries, and Data Processing Operators  
      - **143** – Operators in Data, Accounting, Statistics, Financial Services, and Registry  
      - **144** – Other Administrative Support Staff  
      - **151** – Personal Service Workers  
      - **152** – Sellers  
      - **153** – Personal Care Workers and the Like  
      - **171** – Skilled Construction Workers (except electricians)  
      - **173** – Skilled Workers in Printing, Instrument Making, Jewelry, Crafts, etc.  
      - **175** – Workers in Food Processing, Woodworking, Clothing, and Other Crafts  
      - **191** – Cleaning Workers  
      - **192** – Unskilled Workers in Agriculture, Animal Production, Fisheries, Forestry  
      - **193** – Unskilled Workers in Extractive Industry, Construction, Manufacturing, and Transport  
      - **194** – Meal Preparation Assistants  
      """)

    displaced = categorical_input("Displace", ['Yes', 'No'],
                                  "Is a displaced student?")

    debtor = categorical_input("Debtor", ['Yes', 'No'],
                               "Is student debt tuition?")

    tuition_paid = categorical_input("Tuition fees up to date", ['Yes', 'No'],
                                     "Did student's tuition fees up to date?")

    gender = categorical_input("Gender", ['Male', 'Female'],
                               "Gender of Student")

    scholarship = categorical_input("Scholarship Holder", ['Yes', 'No'],
                                    "Is the student scholarship holder/awardee?")

    prev_grade = st.number_input(
        "Grade of previous qualification", min_value=0.0, max_value=200.0, value=14.0)
    admission_grade = st.number_input(
        "Admission grade", min_value=0.0, max_value=200.0, value=13.0)
    age = st.number_input("Age at enrollment",
                          min_value=15, max_value=80, value=18)
    enrolled_1st = st.number_input(
        "Number of curricular units enrolled by the student in the first semester. ", min_value=0, value=6)
    enrolled_2nd = st.number_input(
        "Number of curricular units enrolled by the student in the second semester. ", min_value=0, value=6)
    unemployment = st.number_input(
        "Unemployement Rate (%)", min_value=0.0, max_value=100.0, value=8.0)
    inflation = st.number_input(
        "Inflation Rate (%)", min_value=0.0, max_value=100.0, value=2.0)
    gdp = st.number_input("GDP", min_value=0.0, value=0.79)

    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    input_data = pd.DataFrame([{
        'Marital_status': marital_status,
        'Application_mode': application_mode,
        'Course': course,
        'Daytime_evening_attendance': attendance,
        'Mothers_qualification': mothers_qualification,
        'Fathers_qualification': fathers_qualification,
        'Mothers_occupation': mothers_occupation,
        'Fathers_occupation': fathers_occupation,
        'Displaced': displaced,
        'Debtor': debtor,
        'Tuition_fees_up_to_date': tuition_paid,
        'Gender': gender,
        'Scholarship_holder': scholarship,
        'Previous_qualification_grade': prev_grade,
        'Admission_grade': admission_grade,
        'Age_at_enrollment': age,
        'Curricular_units_1st_sem_enrolled': enrolled_1st,
        'Curricular_units_2nd_sem_enrolled': enrolled_2nd,
        'Unemployment_rate': unemployment,
        'Inflation_rate': inflation,
        'GDP': gdp
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    prediction_label = "🔴 Dropout" if prediction == 1 else "🟢 No Dropout"
    if proba > 0.5:
        st.warning("⚠️ High risk of Dropout")
    else:
        st.info("✅ Likely to Stay")

    st.success(f"Result: {prediction_label}")
    st.write(f"**Probability of Dropout:** {proba:.2%}")
