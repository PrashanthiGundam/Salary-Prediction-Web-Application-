import pandas as pd 
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\prash\Desktop\10000 coders\job_salary_prediction_dataset.csv")

job_ti=LabelEncoder()
edu_level=LabelEncoder()
cmp_size=LabelEncoder()

df["job_title"]=job_ti.fit_transform(df["job_title"])
df["education_level"]=edu_level.fit_transform(df["education_level"])
df["company_size"]=cmp_size.fit_transform(df["company_size"])

X=df[["job_title","experience_years","education_level","skills_count","company_size"]].values

y=df["salary"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

st.title("salary prediction")
role=st.selectbox("select role",job_ti.classes_)
exp=st.number_input("enter a exp",min_value=0,max_value=30)
edu=st.selectbox("select qfy",edu_level.classes_)
sc=st.number_input("enter your skill count",min_value=0,max_value=30)
cmz=st.selectbox("select size",cmp_size.classes_)

if st.button("predict salary"):

    job=job_ti.fit_transform([role])[0]
    # print(job)
    edu=edu_level.fit_transform([exp])[0]
    cms=cmp_size.fit_transform([edu])[0]
    pred=model.predict([[job,10,edu,5,cms]])

    st.success(round(pred[0],2))