
## Abstact
The air pollution is one of the most significant major driving elements behind climate change and environmental concerns that the Earth is currently facing. Air pollution is a major environmental risk to health. By reducing air pollution levels, countries can reduce many disease such as heart disease, lung cancer, and both chronic and acute respiratory diseases, including asthma. Among the pollutants, particulate matter 2.5 (PM2.5) is the dominant factor causing cancers and other diseases in humans. Several sensors have been positioned in various areas of the world in order to monitor the air quality  but there is still a wide range of areas not covered by those kind of sensors. Trough earth observation projects, Satellites are able to cover a wide range of the Earth's surface, counting inaccessible and hard-to-reach ranges.
  In this project, ground truth data and satellite images  are used to build a multimodal machine learning model to predict PM 2.5 so that it can be applicable where monitoring sensors do not exist. The Best model is the one which use the CNN as backbone for satellite images with a RMSE of .

## Air quality rediction Codes

**Folders and their content can be given below:**

/locations
Contains locations(Geojson file) of the studies area

/air_weather
This folder contains file to create a weather dataset for different ocations by combining the montly hourly data weather data to create a daily dataset for the given location

/data_dir
this folder contains different semtinel images  of N02 and S02 concentration of different locations



/models
 Contains different models and checkpoints

 the first model(weather data is trained for 150 epochs
 the second model with CNN backbone is trained for 200 epochs
 the third model with resnet 34 backbone is trained for 30 epochs
 for the model2 , the best checpoints is 150 whereas for the model3 , the best checkpoint is 25



/extract_S5_images_Nairobi.ipynb
here the mutinmodel daset for Nairobi is created


/extract_S5_images_Nkuru.ipynb
here the mutinmodel daset for Nakuru is created



S5_NO2-1599634901118752-timelapse_Nairobi.gif
S5_NO2-1660064951768767-timelapse_Nakuru.gif
S5_SO2-644910524572098-timelapse_Nakuru.gif
S5_SO2-1465215461745488-timelapse_Nairobi.gif

this four files contain the time lapse images concentrations of SO2 and NO2

Sentinel_5P_CO_Nairobi.csv
Sentinel_5P_CO_nakuru.csv
Sentinel_5P_NO2_Nairobi.csv
Sentinel_5P_NO2_Nakuru.csv
Sentinel_5P_O3_Nairobi.csv
Sentinel_5P_O3_Nakuru.csv

this four files contain the time lapse images concentrations of CO,O3 and NO2 per m2

Sentinel_5P_CO_Nairobi.csv
Sentinel_5P_CO_nakuru.csv
Sentinel_5P_NO2_Nairobi.csv
Sentinel_5P_NO2_Nakuru.csv
Sentinel_5P_O3_Nairobi.csv
Sentinel_5P_O3_Nakuru.csv



train_resnet_backbone.ipynb : this file contain the trainnaing and the evaluation of the model 3
train_single.ipynb: this file contain the trainnaing and the evaluation of the model 1
train.ipynb: this file contain the trainnaing and the evaluation of the model 2

resuts.ipynb: contains the lot of the  results obtained for differents models


## Author :
Name= Armandine Sorel kouyim Meli
email=askmeli@aimsammi.org# Air_quality_prediction
