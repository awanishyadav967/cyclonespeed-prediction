import streamlit as st
#from streamlit_option_menu import option_menu
from PIL import Image
import time
import os
import numpy as np

#import gdown
import torch
from torchvision import transforms




import io
import base64

#import subprocess
#subprocess.run(["pip", "install", "torchvision==0.10.0"])





# --------------------------------------NAV BAR------------------------------------------------------------

#selected = option_menu(
    #menu_title=None,

    #options=["Home", "About", "Contact"],
    #menu_icon="cast",
    #orientation="horizontal",

#)
#if selected == "About":
   # st.title(f"GO to the link for more information-url=https://en.wikipedia.org/wiki/Cyclone#:~:text=In%20meteorology%2C%20a%20cyclone%20(%2F,(opposite%20to%20an%20anticyclone).")
#if selected == "Contact":
   # st.title(f"Email-awanishyadav967@gmail.com")

# ----------------------------------------------Inference script -------------------------------------------------------------

# TODO ADD Python inference script
#PATH = r"./infer/acd_123_34.jpg"

#urla = "https://drive.google.com/uc?id=1XXPduWRnUY582hgfiSddQ2wiz5KR-a0j"
# model_path = r"./model/final_model.ckpt"
#if not os.path.exists("model.pt"):
 #  gdown.download(urla, 'model.pt', quiet=False)

#  upload a file in streamlit
st.header("Predict Cyclone Satellite Image Windspeed")
inp = st.file_uploader("Upload The Cyclone Satellite Image", type=["jpg", "png"])
if inp is not None:
    image = Image.open(io.BytesIO(inp.read())).convert("RGB")
    # inp = r"./infer/acd_123_34.jpg"
    # image = Image.open(inp).convert("RGB")
    test_transforms = transforms.Compose(
        [
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            # All models expect the same normalization mean & std
            # https://pytorch.org/docs/stable/torchvision/models.html
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    image = test_transforms(image)
    image = image.unsqueeze(0)

    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(image)
    output = output.data.squeeze().numpy()

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

    c = st.container()
    c.image(inp, caption='Input Cyclone Image')
   
    c.metric(label="Predicted Wind Speed", value=str(np.round(output, 2)) + " kts")

    i = output

    st.title("Conclusion")

    if i>=10 and i<=40:
        st.subheader("Damage : minimal")
        st.text("No significance structural damage, can uproot trees and cause some flooding in\ncoastal areas.")

    elif i>40 and i<=70:
        st.subheader("Damage : Moderate")
        st.text("No major destruction to buildings, can uproot trees and signs.")
        st.text("Coastal flooding can occur.Secondary effects can include the storage\nof water and electricity.")


    elif(i>70 and i<=100):
        st.subheader("Damage : Extensive")
        st.text("Structural damage to small buildings and serious coastal flooding\nto those on low lying land.")
        st.text("Evacuation may be needed.")


    elif(i>100 and i<=140):
        st.subheader("Damage : Extreme")
        st.text("All sign and trees blown down with extensive damage to roofs.")
        st.text("flats land inland may become flooded.")
        st.text("Evacuation probable.")

    else:
        st.subheader("Damage : Catastrophic")
        st.text("Building destroyed with small buildings being overturned.")
        st.text("All trees and signs blown down.")
        st.text("Evacuation of up to 10 miles inlands")



#-----------------------------------------------------------------------------------------------------------------------


st.title("cyclone speed calculator")

pparticle=st.number_input("Enter the particle density",step=1,min_value=1)
pair =st.number_input("Enter the air density",step=1,min_value=1)
r=st.number_input("Enter the radial distance",step=1,min_value=1)
w=st.number_input("Enter the rotational velocity",step=1,min_value=1)
d=st.number_input("Enter the diameter",step=1,min_value=1)
u=st.number_input("Enter the air viscosity",step=1,min_value=1)

p=(pparticle-pair)
radialvelocity=r*w*w*d*d*p/(18*u)

st.success(f"The radial velocity is {radialvelocity}")






# -----------------------------------------------------Alert Pdf---------------------------------------



st.header("Official Alerts")


#This part is for displaying pdf from local using html embeded or ifram
filename = 'test.pdf'

with open(filename, "rb") as f:
     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

     # Embedding PDF in HTML
     pdf_display = F'<center><embed src="data:application/pdf;base64,{base64_pdf}" width="400" height="500" type="application/pdf"></center>'
     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
      # Displaying File
     st.markdown(pdf_display, unsafe_allow_html=True)


#-------------------------------------------------------------------------------------------------------------------
