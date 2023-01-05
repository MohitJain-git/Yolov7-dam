import streamlit as st
import pandas as pd
import numpy as np
from  PIL import Image
# from io import StringIO 
from check import *
import os

def save_uploadedfile(uploadedfile):
     with open(os.path.join("uploaded",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("File is successfully uploaded")

st.title("Demo for Image Processing")
file = st.file_uploader("Please choose a file",type=['jpg','png','jpeg'])
data = []

if file is not None:
    image=Image.open(file)
    save_uploadedfile(file)
    data=detect(source=os.path.join("uploaded",file.name))
    image_dir=os.path.join(data[0],file.name)
    if(len(data[1][0]) > 1):
        st.write("All persons from the left to right are:")
    for i in data[1][0]:
        x = "attr"
        i[1] = float(i[1])*100 
        i[1] = f'{i[1]:.2f}'
        if i[3]["attr"] != []:
            st.write(f"**{i[0]}** detected with Confidence: {float(i[1])}% wearing {str(i[3][x])[1:-1]}")
        else:
            st.write(f"**{i[0]}** detected with Confidence: {float(i[1])}%")
    st.image(image_dir)

    # st.image(image)
# st.image("../yolov7/inference/images/bus.jpg")