# SELECTIVE ANOMALY DETECTION IN SURVEILLANCE VIDEOS
This project is aimed at building a practical anomaly detection and live streaming application that utilizes a Deep MIL ranking model via transfer learning.

## DEPENDENCIES
All software dependencies are listed in the `requirements.txt` file. Navigate to `project_app` and execute:
```shell
pip install -r requirements.txt
```

## DOCKER IMAGE
We also provide a dockerfile for building and running a virtual instance of the application. Navigate to `kami_no_me` and follow the instructions in `readme.md`

## LIVE STREAM DETECTION
For live stream detection navigate to `real-time` and run `view.py`

## DATASET
UCF-Crime Dataset (https://www.crcv.ucf.edu/projects/real-world/) is used for evaluation.
You can find preprocessed dataset features on the `features` branch of the repository

## BASELINE IMPLEMENTATION
The implementation of the Deep MIL Ranking model was strongly based on these previous works:
- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
- https://github.com/ptirupat/AnomalyDetection_CVPR18
- https://github.com/adamcasson/c3d
- https://github.com/fluque1995/tfm-anomaly-detection
