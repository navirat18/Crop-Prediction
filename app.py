import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import gradio as gr




data=pd.read_csv('cpdata.csv')


x=data.drop("label",axis=1)
y=data["label"]
LE=LabelEncoder()
y_encoded=LE.fit_transform(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y_encoded,test_size=0.2,random_state=0)



classi=RandomForestClassifier()
classi.fit(xtrain,ytrain)


def laa(temparature,humidity,ph,rainfall,N,P,K):
    dd = classi.predict([[temparature,humidity,ph,rainfall,N,P,K]])
    ss = LE.inverse_transform(dd)[0]
    return f"The crops uh need to plant is : {ss}"

# Gradio interface
interface = gr.Interface(
    fn=laa,
    inputs=[
        gr.Number(label="Temparature", value=22),
        gr.Number(label="Humidity", value=1200),
        gr.Number(label="ph", value=12),
        gr.Number(label="rainfall", value=700),
        gr.Number(label="N", value=5),
        gr.Number(label="P", value=5),
        gr.Number(label="K", value=5)
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Crops Prediction ",
    description="Enter the crops details to predict the correxct crop"
)

interface.launch(inline=True)