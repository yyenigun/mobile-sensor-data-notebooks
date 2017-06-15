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

# Loading Gyroscope Data

drivingCarGyroFiles = glob.glob(drivingCarPath + "/0_Gyroscope*.csv")
drivingCarGyroDf = pd.DataFrame()
list_ = []
for file_ in drivingCarGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarGyroDf = pd.concat(list_)
drivingCarGyroDf['label'] = 'driving car'

transitGyroFiles = glob.glob(transitPath + "/0_Gyroscope*.csv")
transitGyroDf = pd.DataFrame()
list_ = []
for file_ in transitGyroFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
transitGyroDf = pd.concat(list_)
transitGyroDf['label'] = 'transit'

gyroDfFrames = [drivingCarGyroDf, transitGyroDf]

gyroDf = pd.concat(gyroDfFrames)

# Windowing Magnetometer Data

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

# Merge Features

accMagFeatures['timestamps'] = accMagFeatures.index
gyroFeatures['timestamps'] = gyroFeatures.index

accMagFeatures = accMagFeatures.sort_values(by='timestamps')
gyroFeatures = gyroFeatures.sort_values(by='timestamps')

allFeatures = pd.merge_asof(accMagFeatures, gyroFeatures, on='timestamps', by='label',
                            tolerance=pd.Timedelta('100ms'))

# Remove Null Features

print(len(accMagFeatures.index))
print(len(gyroFeatures.index))
print(len(allFeatures.index))

allFeatures = allFeatures[~(allFeatures.gyro_mag_mean.isnull()) & ~(allFeatures.acc_mag_mean.isnull())]

len(allFeatures.index)

# Run Classification For Magnetometer & Accelerometer
h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(allFeatures)

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
    'gyro_z_max',
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
    model_id="DrivingTransitAccelerometerGyroscope",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
