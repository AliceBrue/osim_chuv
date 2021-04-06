# osim_chuv

This repository contains python scripts to evaluate osim upper limb models from kinematics and EMG data.
'emg' and 'kinematics' folders contain EMG and kinematics data, 'kinematics' folder also contains setup files for osim CMC tool, 
'models' folder contains osim models.
Python scripts are the following:
- emg_control.py: simulate osim model with EMG data of a movement of interest as muscle activation and compare the resulting kinematics with references data
- kinematics.py: scale kinematics data of a movement of interest and compare osim IK and CMC results to references and EMG data
- opensim_environement.py: opensim python environement
- osim_model.py: functions to modify muscles and joints of osim model
