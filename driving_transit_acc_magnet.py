import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Accelerometer Data

drivingCarPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Driving Car/csv'
drivingCarFiles = glob.glob(drivingCarPath + "/0_Accelerometer*.csv")
accDfCar = pd.DataFrame()
list_ = []
for file_ in drivingCarFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfCar = pd.concat(list_)
accDfCar['label'] = 'driving car'

transitPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Transit/csv'
transitFiles = glob.glob(transitPath + "/0_Accelerometer*.csv")
accDfTransit = pd.DataFrame()
list_ = []
for file_ in transitFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfTransit = pd.concat(list_)
accDfTransit['label'] = 'transit'

accDfFrames = [accDfCar, accDfTransit]

accDf = pd.concat(accDfFrames)

xV = (accDf['x'] * accDf['x']) + (accDf['y'] * accDf['y']) + (accDf['z'] * accDf['z'])
accDf['acc_mag'] = xV
accDf['acc_mag'] = accDf['acc_mag'].apply(math.sqrt)

# Convert timestamp to date time

accDf['start'] = pd.to_datetime(accDf['start'])
accDf['end'] = pd.to_datetime(accDf['end'])

print(len(accDf.index))

# Windowing Accelerometer Data

accMagFeatures = pd.DataFrame()

accDf['timestamps'] = pd.to_datetime(accDf['timestamps'])
accDf.set_index(['timestamps'])
accDf = accDf.sort_values(by='timestamps')

accMagDf = accDf[['timestamps', 'start', 'acc_mag', 'x', 'y', 'z', 'label']]
accMagDf = accMagDf.set_index(['timestamps'])

accMagFeatures['acc_mag_mean'] = accMagDf['acc_mag'].rolling('1s').mean()
accMagFeatures['acc_mag_std'] = accMagDf['acc_mag'].rolling('1s').std()
accMagFeatures['acc_mag_var'] = accMagDf['acc_mag'].rolling('1s').var()
accMagFeatures['acc_mag_min'] = accMagDf['acc_mag'].rolling('1s').min()
accMagFeatures['acc_mag_max'] = accMagDf['acc_mag'].rolling('1s').max()
accMagFeatures['acc_x_mean'] = accMagDf['x'].rolling('1s').mean()
accMagFeatures['acc_x_std'] = accMagDf['x'].rolling('1s').std()
accMagFeatures['acc_x_var'] = accMagDf['x'].rolling('1s').var()
accMagFeatures['acc_x_min'] = accMagDf['x'].rolling('1s').min()
accMagFeatures['acc_x_max'] = accMagDf['x'].rolling('1s').max()
accMagFeatures['acc_y_mean'] = accMagDf['y'].rolling('1s').mean()
accMagFeatures['acc_y_std'] = accMagDf['y'].rolling('1s').std()
accMagFeatures['acc_y_var'] = accMagDf['y'].rolling('1s').var()
accMagFeatures['acc_y_min'] = accMagDf['y'].rolling('1s').min()
accMagFeatures['acc_y_max'] = accMagDf['y'].rolling('1s').max()
accMagFeatures['acc_z_mean'] = accMagDf['z'].rolling('1s').mean()
accMagFeatures['acc_z_std'] = accMagDf['z'].rolling('1s').std()
accMagFeatures['acc_z_var'] = accMagDf['z'].rolling('1s').var()
accMagFeatures['acc_z_min'] = accMagDf['z'].rolling('1s').min()
accMagFeatures['acc_z_max'] = accMagDf['z'].rolling('1s').max()
accMagFeatures['label'] = accMagDf['label']

# Loading Magnetometer Data

drivingCarMagFiles = glob.glob(drivingCarPath + "/0_Magnetometer*.csv")
drivingCarMagDf = pd.DataFrame()
list_ = []
for file_ in drivingCarMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarMagDf = pd.concat(list_)
drivingCarMagDf['label'] = 'driving car'

transitMagFiles = glob.glob(transitPath + "/0_Magnetometer*.csv")
transitMagDf = pd.DataFrame()
list_ = []
for file_ in transitMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
transitMagDf = pd.concat(list_)
transitMagDf['label'] = 'transit'

magDfFrames = [drivingCarMagDf, transitMagDf]

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

accMagFeatures['timestamps'] = accMagFeatures.index
magFeatures['timestamps'] = magFeatures.index

accMagFeatures = accMagFeatures.sort_values(by='timestamps')
magFeatures = magFeatures.sort_values(by='timestamps')

allFeatures = pd.merge_asof(accMagFeatures, magFeatures, on='timestamps', by='label',
                            tolerance=pd.Timedelta('100ms'))

# Remove Null Features

print(len(accMagFeatures.index))
print(len(magFeatures.index))
print(len(allFeatures.index))

allFeatures = allFeatures[~(allFeatures.mag_mag_mean.isnull()) & ~(allFeatures.acc_mag_mean.isnull())]

len(allFeatures.index)

# Run Classification For Magnetometer & Accelerometer
h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(allFeatures)

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
    'mag_z_max',
    'acc_mag_mean',
    'acc_mag_std',
    'acc_mag_var',
    'acc_mag_min',
    'acc_mag_max',
    'acc_x_mean',
    'acc_x_std',
    'acc_x_var',
    'acc_x_min',
    'acc_x_max',
    'acc_y_mean',
    'acc_y_std',
    'acc_y_var',
    'acc_y_min',
    'acc_y_max',
    'acc_z_mean',
    'acc_z_std',
    'acc_z_var',
    'acc_z_min',
    'acc_z_max'

]

random_forest_model = h2o.H2ORandomForestEstimator(
    model_id="DrivingTransitAccelerometerMagnetometer",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
