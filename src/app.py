
# In[5]:


import streamlit as st
import pickle
import numpy as np
with open('../model/plank_dp.pkl', 'rb') as file:
    model = pickle.load(file)
    


# In[4]:


st.title("Plank correction prediction")


# In[6]:


# Input fields for all features
nose_x = st.number_input("Nose X", value=0.0)
nose_y = st.number_input("Nose Y", value=0.0)
nose_z = st.number_input("Nose Z", value=0.0)
nose_v = st.number_input("Nose V", value=1.0)

left_shoulder_x = st.number_input("Left Shoulder X", value=0.0)
left_shoulder_y = st.number_input("Left Shoulder Y", value=0.0)
left_shoulder_z = st.number_input("Left Shoulder Z", value=0.0)
left_shoulder_v = st.number_input("Left Shoulder V", value=1.0)

right_shoulder_x = st.number_input("Right Shoulder X", value=0.0)
right_shoulder_y = st.number_input("Right Shoulder Y", value=0.0)
right_shoulder_z = st.number_input("Right Shoulder Z", value=0.0)
right_shoulder_v = st.number_input("Right Shoulder V", value=1.0)

left_elbow_x = st.number_input("Left Elbow X", value=0.0)
left_elbow_y = st.number_input("Left Elbow Y", value=0.0)
left_elbow_z = st.number_input("Left Elbow Z", value=0.0)
left_elbow_v = st.number_input("Left Elbow V", value=1.0)

right_elbow_x = st.number_input("Right Elbow X", value=0.0)
right_elbow_y = st.number_input("Right Elbow Y", value=0.0)
right_elbow_z = st.number_input("Right Elbow Z", value=0.0)
right_elbow_v = st.number_input("Right Elbow V", value=1.0)

left_wrist_x = st.number_input("Left Wrist X", value=0.0)
left_wrist_y = st.number_input("Left Wrist Y", value=0.0)
left_wrist_z = st.number_input("Left Wrist Z", value=0.0)
left_wrist_v = st.number_input("Left Wrist V", value=1.0)

right_wrist_x = st.number_input("Right Wrist X", value=0.0)
right_wrist_y = st.number_input("Right Wrist Y", value=0.0)
right_wrist_z = st.number_input("Right Wrist Z", value=0.0)
right_wrist_v = st.number_input("Right Wrist V", value=1.0)

left_hip_x = st.number_input("Left Hip X", value=0.0)
left_hip_y = st.number_input("Left Hip Y", value=0.0)
left_hip_z = st.number_input("Left Hip Z", value=0.0)
left_hip_v = st.number_input("Left Hip V", value=1.0)

right_hip_x = st.number_input("Right Hip X", value=0.0)
right_hip_y = st.number_input("Right Hip Y", value=0.0)
right_hip_z = st.number_input("Right Hip Z", value=0.0)
right_hip_v = st.number_input("Right Hip V", value=1.0)

left_knee_x = st.number_input("Left Knee X", value=0.0)
left_knee_y = st.number_input("Left Knee Y", value=0.0)
left_knee_z = st.number_input("Left Knee Z", value=0.0)
left_knee_v = st.number_input("Left Knee V", value=1.0)

right_knee_x = st.number_input("Right Knee X", value=0.0)
right_knee_y = st.number_input("Right Knee Y", value=0.0)
right_knee_z = st.number_input("Right Knee Z", value=0.0)
right_knee_v = st.number_input("Right Knee V", value=1.0)

left_ankle_x = st.number_input("Left Ankle X", value=0.0)
left_ankle_y = st.number_input("Left Ankle Y", value=0.0)
left_ankle_z = st.number_input("Left Ankle Z", value=0.0)
left_ankle_v = st.number_input("Left Ankle V", value=1.0)

right_ankle_x = st.number_input("Right Ankle X", value=0.0)
right_ankle_y = st.number_input("Right Ankle Y", value=0.0)
right_ankle_z = st.number_input("Right Ankle Z", value=0.0)
right_ankle_v = st.number_input("Right Ankle V", value=1.0)

left_heel_x = st.number_input("Left Heel X", value=0.0)
left_heel_y = st.number_input("Left Heel Y", value=0.0)
left_heel_z = st.number_input("Left Heel Z", value=0.0)
left_heel_v = st.number_input("Left Heel V", value=1.0)

right_heel_x = st.number_input("Right Heel X", value=0.0)
right_heel_y = st.number_input("Right Heel Y", value=0.0)
right_heel_z = st.number_input("Right Heel Z", value=0.0)
right_heel_v = st.number_input("Right Heel V", value=1.0)

left_foot_index_x = st.number_input("Left Foot Index X", value=0.0)
left_foot_index_y = st.number_input("Left Foot Index Y", value=0.0)
left_foot_index_z = st.number_input("Left Foot Index Z", value=0.0)
left_foot_index_v = st.number_input("Left Foot Index V", value=1.0)

right_foot_index_x = st.number_input("Right Foot Index X", value=0.0)
right_foot_index_y = st.number_input("Right Foot Index Y", value=0.0)
right_foot_index_z = st.number_input("Right Foot Index Z", value=0.0)
right_foot_index_v = st.number_input("Right Foot Index V", value=1.0)


# In[7]:


# Collect input into a single list
input_data = np.array([
    nose_x, nose_y, nose_z, nose_v,
    left_shoulder_x, left_shoulder_y, left_shoulder_z, left_shoulder_v,
    right_shoulder_x, right_shoulder_y, right_shoulder_z, right_shoulder_v,
    left_elbow_x, left_elbow_y, left_elbow_z, left_elbow_v,
    right_elbow_x, right_elbow_y, right_elbow_z, right_elbow_v,
    left_wrist_x, left_wrist_y, left_wrist_z, left_wrist_v,
    right_wrist_x, right_wrist_y, right_wrist_z, right_wrist_v,
    left_hip_x, left_hip_y, left_hip_z, left_hip_v,
    right_hip_x, right_hip_y, right_hip_z, right_hip_v,
    left_knee_x, left_knee_y, left_knee_z, left_knee_v,
    right_knee_x, right_knee_y, right_knee_z, right_knee_v,
    left_ankle_x, left_ankle_y, left_ankle_z, left_ankle_v,
    right_ankle_x, right_ankle_y, right_ankle_z, right_ankle_v,
    left_heel_x, left_heel_y, left_heel_z, left_heel_v,
    right_heel_x, right_heel_y, right_heel_z, right_heel_v,
    left_foot_index_x, left_foot_index_y, left_foot_index_z, left_foot_index_v,
    right_foot_index_x, right_foot_index_y, right_foot_index_z, right_foot_index_v
])


# In[8]:


def predict_posture(features):
    features = features.reshape(1, -1)  # Reshape for the model input
    prediction = model.predict(features)  # Predict using the loaded model
    return prediction


# In[9]:


if st.button('Predict'):
    result = predict_posture(input_data)
    st.write(f"The predicted posture is: {result}")


# In[ ]:




