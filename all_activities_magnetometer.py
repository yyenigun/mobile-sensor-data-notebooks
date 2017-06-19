import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Magnetometer Data
drivingCarPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Driving Car/csv'
drivingCarMagFiles = glob.glob(drivingCarPath + "/0_Magnetometer*.csv")
drivingCarMagDf = pd.DataFrame()
list_ = []
for file_ in drivingCarMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarMagDf = pd.concat(list_)
drivingCarMagDf['label'] = 'driving car'

busPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Bus*'
busMagFiles = glob.glob(busPath + "/trim_0_Magnetometer*.csv")
busMagDf = pd.DataFrame()
list_ = []
for file_ in busMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
busMagDf = pd.concat(list_)
busMagDf['label'] = 'bus'

lightRailPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Light_Rail*'
lightRailMagFiles = glob.glob(lightRailPath + "/trim_0_Magnetometer*.csv")
lightRailMagDf = pd.DataFrame()
list_ = []
for file_ in lightRailMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
lightRailMagDf = pd.concat(list_)
lightRailMagDf['label'] = 'light rail'

standingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Standing*'
standingMagFiles = glob.glob(standingPath + "/trim_0_Magnetometer*.csv")
standingMagDf = pd.DataFrame()
list_ = []
for file_ in standingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
standingMagDf = pd.concat(list_)
standingMagDf['label'] = 'standing'

walkingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Walking*'
walkingMagFiles = glob.glob(walkingPath + "/trim_0_Magnetometer*.csv")
walkingMagDf = pd.DataFrame()
list_ = []
for file_ in walkingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
walkingMagDf = pd.concat(list_)
walkingMagDf['label'] = 'walking'

eatingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Eating-Drinking/csv'
eatingMagFiles = glob.glob(eatingPath + "/0_Magnetometer*.csv")
eatingMagDf = pd.DataFrame()
list_ = []
for file_ in eatingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingMagDf = pd.concat(list_)
eatingMagDf['label'] = 'eating'

elevatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Elevator*'
elevatorMagFiles = glob.glob(elevatorPath + "/trim_0_Magnetometer*.csv")
elevatorMagDf = pd.DataFrame()
list_ = []
for file_ in elevatorMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorMagDf = pd.concat(list_)
elevatorMagDf['label'] = 'elevator'

escalatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Escalator*'
escalatorMagFiles = glob.glob(escalatorPath + "/trim_0_Magnetometer*.csv")
escalatorMagDf = pd.DataFrame()
list_ = []
for file_ in escalatorMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
escalatorMagDf = pd.concat(list_)
escalatorMagDf['label'] = 'escalator'

magDfFrames = [drivingCarMagDf, busMagDf, lightRailMagDf, walkingMagDf, eatingMagDf, elevatorMagDf, escalatorMagDf]

magDf = pd.concat(magDfFrames)

# Windowing Magnetometer Data

magV = (magDf['x'] * magDf['x']) + (magDf['y'] * magDf['y']) + (magDf['z'] * magDf['z'])
magDf['mag_magnitude'] = magV
magDf['mag_magnitude'] = magDf['mag_magnitude'].apply(math.sqrt)

# Convert timestamp to date time

magDf['start'] = pd.to_datetime(magDf['start'])
magDf['end'] = pd.to_datetime(magDf['end'])

print(len(magDf.index))

magFeatures = pd.DataFrame()

magDf['timestamps'] = pd.to_datetime(magDf['timestamps'])
magDf.set_index(['timestamps'])
magDf = magDf.sort_values(by='timestamps')

magDf = magDf[['timestamps', 'start', 'mag_magnitude', 'x', 'y', 'z', 'label']]
magDf = magDf.set_index(['timestamps'])

magFeatures['mag_mag_mean'] = magDf['mag_magnitude'].rolling('1s').mean()
magFeatures['mag_mag_std'] = magDf['mag_magnitude'].rolling('1s').std()
magFeatures['mag_mag_var'] = magDf['mag_magnitude'].rolling('1s').var()
magFeatures['mag_mag_min'] = magDf['mag_magnitude'].rolling('1s').min()
magFeatures['mag_mag_max'] = magDf['mag_magnitude'].rolling('1s').max()
magFeatures['mag_x_mean'] = magDf['x'].rolling('1s').mean()
magFeatures['mag_x_std'] = magDf['x'].rolling('1s').std()
magFeatures['mag_x_var'] = magDf['x'].rolling('1s').var()
magFeatures['mag_x_min'] = magDf['x'].rolling('1s').min()
magFeatures['mag_x_max'] = magDf['x'].rolling('1s').max()
magFeatures['mag_y_mean'] = magDf['y'].rolling('1s').mean()
magFeatures['mag_y_std'] = magDf['y'].rolling('1s').std()
magFeatures['mag_y_var'] = magDf['y'].rolling('1s').var()
magFeatures['mag_y_min'] = magDf['y'].rolling('1s').min()
magFeatures['mag_y_max'] = magDf['y'].rolling('1s').max()
magFeatures['mag_z_mean'] = magDf['z'].rolling('1s').mean()
magFeatures['mag_z_std'] = magDf['z'].rolling('1s').std()
magFeatures['mag_z_var'] = magDf['z'].rolling('1s').var()
magFeatures['mag_z_min'] = magDf['z'].rolling('1s').min()
magFeatures['mag_z_max'] = magDf['z'].rolling('1s').max()
magFeatures['label'] = magDf['label']

# Merge Features

magFeatures['timestamps'] = magFeatures.index
magFeatures = magFeatures.sort_values(by='timestamps')

# Run Classification For Magnetometer & Accelerometer
h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(magFeatures)

continuous_feature_columns = [
    'mag_mag_mean',
    'mag_mag_std',
    'mag_mag_var',
    'mag_mag_min',
    'mag_mag_max',
    'mag_x_mean',
    'mag_x_std',
    'mag_x_var',
    'mag_x_min',
    'mag_x_max',
    'mag_y_mean',
    'mag_y_std',
    'mag_y_var',
    'mag_y_min',
    'mag_y_max',
    'mag_z_mean',
    'mag_z_std',
    'mag_z_var',
    'mag_z_min',
    'mag_z_max'

]

random_forest_model = h2o.H2ORandomForestEstimator(
    model_id="AllActivitiesMagnetometer",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
