import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Accelerometer Data for Driving Car, Bus, Light Rail, Standing, Walking

drivingCarPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Driving Car/csv'
busPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Bus*'
lightRailPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Light_Rail*'
standingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Standing*'
walkingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Walking*'
eatingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Eating-Drinking/csv'
elevatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Elevator*'
escalatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Escalator*'

# Loading Gyroscope Data
drivingCarGyroFiles = glob.glob(drivingCarPath + "/0_Gyroscope*.csv")
drivingCarGyroDf = pd.DataFrame()
list_ = []
for file_ in drivingCarGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarGyroDf = pd.concat(list_)
drivingCarGyroDf['label'] = 'driving car'

busGyroFiles = glob.glob(busPath + "/trim_0_Gyroscope*.csv")
busGyroDf = pd.DataFrame()
list_ = []
for file_ in busGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
busGyroDf = pd.concat(list_)
busGyroDf['label'] = 'bus'

lightRailGyroFiles = glob.glob(lightRailPath + "/trim_0_Gyroscope*.csv")
lightRailGyroDf = pd.DataFrame()
list_ = []
for file_ in lightRailGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
lightRailGyroDf = pd.concat(list_)
lightRailGyroDf['label'] = 'light rail'

standingGyroFiles = glob.glob(standingPath + "/trim_0_Gyroscope*.csv")
standingGyroDf = pd.DataFrame()
list_ = []
for file_ in standingGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
standingGyroDf = pd.concat(list_)
standingGyroDf['label'] = 'standing'

walkingGyroFiles = glob.glob(walkingPath + "/trim_0_Gyroscope*.csv")
walkingGyroDf = pd.DataFrame()
list_ = []
for file_ in walkingGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
walkingGyroDf = pd.concat(list_)
walkingGyroDf['label'] = 'walking'

eatingGyroFiles = glob.glob(eatingPath + "/0_Gyroscope*.csv")
eatingGyroDf = pd.DataFrame()
list_ = []
for file_ in eatingGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingGyroDf = pd.concat(list_)
eatingGyroDf['label'] = 'eating'

elevatorGyroFiles = glob.glob(elevatorPath + "/trim_0_Gyroscope*.csv")
elevatorGyroDf = pd.DataFrame()
list_ = []
for file_ in elevatorGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorGyroDf = pd.concat(list_)
elevatorGyroDf['label'] = 'elevator'

escalatorGyroFiles = glob.glob(escalatorPath + "/trim_0_Gyroscope*.csv")
escalatorGyroDf = pd.DataFrame()
list_ = []
for file_ in escalatorGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
escalatorGyroDf = pd.concat(list_)
escalatorGyroDf['label'] = 'escalator'

gyroDfFrames = [drivingCarGyroDf, busGyroDf, lightRailGyroDf, standingGyroDf, walkingGyroDf, eatingGyroDf, elevatorGyroDf, escalatorGyroDf]

gyroDf = pd.concat(gyroDfFrames)

# Windowing Gyroscope Magnitude Data

gyroV = (gyroDf['x'] * gyroDf['x']) + (gyroDf['y'] * gyroDf['y']) + (gyroDf['z'] * gyroDf['z'])
gyroDf['gyro_magnitude'] = gyroV
gyroDf['gyro_magnitude'] = gyroDf['gyro_magnitude'].apply(math.sqrt)

# Convert timestamp to date time

gyroDf['start'] = pd.to_datetime(gyroDf['start'])
gyroDf['end'] = pd.to_datetime(gyroDf['end'])

print(len(gyroDf.index))

gyroFeatures = pd.DataFrame()

gyroDf['timestamps'] = pd.to_datetime(gyroDf['timestamps'])
gyroDf.set_index(['timestamps'])
gyroDf = gyroDf.sort_values(by='timestamps')

gyroDf = gyroDf[['timestamps', 'start', 'gyro_magnitude', 'x', 'y', 'z', 'label']]
gyroDf = gyroDf.set_index(['timestamps'])

# Feature extraction

gyroFeatures['gyro_mag_mean'] = gyroDf['gyro_magnitude'].rolling('1s').mean()
gyroFeatures['gyro_mag_std'] = gyroDf['gyro_magnitude'].rolling('1s').std()
gyroFeatures['gyro_mag_var'] = gyroDf['gyro_magnitude'].rolling('1s').var()
gyroFeatures['gyro_mag_min'] = gyroDf['gyro_magnitude'].rolling('1s').min()
gyroFeatures['gyro_mag_max'] = gyroDf['gyro_magnitude'].rolling('1s').max()
gyroFeatures['gyro_x_mean'] = gyroDf['x'].rolling('1s').mean()
gyroFeatures['gyro_x_std'] = gyroDf['x'].rolling('1s').std()
gyroFeatures['gyro_x_var'] = gyroDf['x'].rolling('1s').var()
gyroFeatures['gyro_x_min'] = gyroDf['x'].rolling('1s').min()
gyroFeatures['gyro_x_max'] = gyroDf['x'].rolling('1s').max()
gyroFeatures['gyro_y_mean'] = gyroDf['y'].rolling('1s').mean()
gyroFeatures['gyro_y_std'] = gyroDf['y'].rolling('1s').std()
gyroFeatures['gyro_y_var'] = gyroDf['y'].rolling('1s').var()
gyroFeatures['gyro_y_min'] = gyroDf['y'].rolling('1s').min()
gyroFeatures['gyro_y_max'] = gyroDf['y'].rolling('1s').max()
gyroFeatures['gyro_z_mean'] = gyroDf['z'].rolling('1s').mean()
gyroFeatures['gyro_z_std'] = gyroDf['z'].rolling('1s').std()
gyroFeatures['gyro_z_var'] = gyroDf['z'].rolling('1s').var()
gyroFeatures['gyro_z_min'] = gyroDf['z'].rolling('1s').min()
gyroFeatures['gyro_z_max'] = gyroDf['z'].rolling('1s').max()
gyroFeatures['label'] = gyroDf['label']

# Run Classification For Pressure & Accelerometer

h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(gyroFeatures)

continuous_feature_columns = [
    'gyro_mag_mean',
    'gyro_mag_std',
    'gyro_mag_var',
    'gyro_mag_min',
    'gyro_mag_max',
    'gyro_x_mean',
    'gyro_x_std',
    'gyro_x_var',
    'gyro_x_min',
    'gyro_x_max',
    'gyro_y_mean',
    'gyro_y_std',
    'gyro_y_var',
    'gyro_y_min',
    'gyro_y_max',
    'gyro_z_mean',
    'gyro_z_std',
    'gyro_z_var',
    'gyro_z_min',
    'gyro_z_max'
]

random_forest_model = h2o.H2ORandomForestEstimator(
    model_id="AllActivitiesGyroscope",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
