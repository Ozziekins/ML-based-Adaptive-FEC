# Advanced Machine Learning Project

## Adpative FEC for Cloud Gaming Service

This repository contains the project implementation for the Advanced Machine Learning course, focusing on Adaptive Forward Error Correction (FEC) for a cloud gaming service called LoudPlay.

### Background
Cloud gaming is a technology that allows people to play games on remote servers while streaming the action to their devices via the internet. This eliminates the need for high-end gaming gear and allows consumers to access and play games on a variety of devices, such as smartphones, tablets, and low-spec laptops. However, cloud gaming is heavily dependent on the quality and stability of the network connection, as any network disruptions or delays can result in a bad gaming experience with difficulties such as input latency and video distortions.

Forward error correction (FEC) is a method of obtaining error control in data transmission in which the source (transmitter) sends redundant data and the destination (receiver) recognizes only the portion of the data that contains no apparent errors [[1](https://www.techtarget.com/searchmobilecomputing/definition/forward-error-correction#:~:text=Forward%20error%20correction%20(FEC)%20is,that%20contains%20no%20apparent%20errors.)]. Traditional FEC schemes use a fixed level of redundancy, which may not be optimal for varying network conditions, such as fluctuations in packet loss or latency.

Adaptive FEC adapts the level of redundancy in real-time based on the observed network conditions. By dynamically adjusting the FEC parameters, adaptive FEC can provide better error recovery and improve the quality of service for cloud gaming.

### Project Description
The goal of this project is to implement and evaluate an Adaptive FEC solution for the LoudPlay cloud gaming service. The project consists of several components and folders, which are described below:

- `backend`: This folder contains the implementation of the backend components for the cloud gaming service.
- `Clustering notebooks`: This folder contains different versions of the clustering notebooks used during the project.
- `dataset`: This folder contains the zipped dataset used and html files presenting the Pandas profiling result of the dataset.
- `pdf`: This folder includes the project proposal, reference paper, presentation slides, and the final project report.
- `streamlit`: This folder contains the Streamlit application for the project's demo interface.
- `.gitignore`: This file specifies the files and directories to be ignored by version control systems.
- `requirements.txt`: This file specifies the dependencies and libraries for the project
- `Regression.ipynb`, `solution-2.ipynb`, `solution.ipynb`: These Jupyter Notebook files contain different versions of the regression models and solutions developed during the project for comparison and evaluation purposes.

### Getting Started
To get started with the LoudPlay Adaptive FEC for Cloud Gaming Service project, follow these steps:

1. Clone the repository to your local machine.
2. Create and activate a virtual environment using: 
    - `python3 -m venv project`.  
    - `source project/bin/aactivate`
3. Install the required dependencies and libraries specified in the project's requirements.txt file using: 
    - `pip3 install -r requirements.txt`.
4. Start the server using: 
    - `uvicorn backend.app:app --reload`.
5. On another terminal, start the streamlit application using: 
    - `streamlit run streamlit/home.py --server.maxUploadSize=500`.

> If desired, refer to the provided documentation, including the project proposal, reference paper, presentation slides, and project report, for detailed information on the project's background, objectives, and findings.

### Contributors
The Adaptive FEC for Cloud Gaming Service project was developed by [@CodeSmith](https://github.com/AbdulmueezEmiola) [@Ozzie_kins](https://github.com/Ozziekins) [@Pain122](https://github.com/Pain122) as part of the Advanced Machine Learning course. Contributions were made by each team member, including research, implementation