import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Accelerometer Data

eatingPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Eating-Drinking/csv'
eatingFiles = glob.glob(eatingPath + "/0_Accelerometer*.csv")
accDfEating = pd.DataFrame()
list_ = []
for file_ in eatingFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfEating = pd.concat(list_)
accDfEating['label'] = 'eating'

elevatorPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Elevator-Escalator/csv-segments/dir_Elevator*'
elevatorFiles = glob.glob(elevatorPath + "/trim_0_Accelerometer*.csv")
accDfElevator = pd.DataFrame()
list_ = []
for file_ in elevatorFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfElevator = pd.concat(list_)
accDfElevator['label'] = 'elevator'

escalatorPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Elevator-Escalator/csv-segments/dir_Escalator*'
escalatorFiles = glob.glob(escalatorPath + "/trim_0_Accelerometer*.csv")
accDfEscalator = pd.DataFrame()
list_ = []
for file_ in escalatorFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfEscalator = pd.concat(list_)
accDfEscalator['label'] = 'escalator'

accDfFrames = [accDfEating, accDfElevator, accDfEscalator]

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

len(accMagFeatures.index)


# Loading Barometer Data From Smart Watch

eatingBarFiles = glob.glob(eatingPath + "/0_MsBandBarometer*.csv")
eatingBarDf = pd.DataFrame()
list_ = []
for file_ in eatingBarFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingBarDf = pd.concat(list_)
eatingBarDf['label'] = 'eating'

elevatorBarPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Elevator-Escalator/csv-segments/dir_Elevator*'
elevatorBarFiles = glob.glob(elevatorBarPath + "/trim_0_MsBandBarometer*.csv")
elevatorBarDf = pd.DataFrame()
list_ = []
for file_ in elevatorBarFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorBarDf = pd.concat(list_)
elevatorBarDf['label'] = 'elevator'

escalatorBarPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Elevator-Escalator/csv-segments/dir_Escalator*'
escalatorBarFiles = glob.glob(escalatorBarPath + "/trim_0_MsBandBarometer*.csv")
escalatorBarDf = pd.DataFrame()
list_ = []
for file_ in escalatorBarFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
escalatorBarDf = pd.concat(list_)
escalatorBarDf['label'] = 'escalator'

barometerDfFrames = [eatingBarDf, elevatorBarDf, escalatorBarDf]

barometerDf = pd.concat(barometerDfFrames)

# Windowing Gyroscope Magnitude Data

# Convert timestamp to date time

barometerDf['start'] = pd.to_datetime(barometerDf['start'])
barometerDf['end'] = pd.to_datetime(barometerDf['end'])

print(len(barometerDf.index))

barometerFeatures = pd.DataFrame()

barometerDf['timestamps'] = pd.to_datetime(barometerDf['timestamps'])
barometerDf.set_index(['timestamps'])
barometerDf = barometerDf.sort_values(by='timestamps')

barometerDf = barometerDf[['timestamps', 'start', 'airPressure', 'temperature', 'label']]
barometerDf = barometerDf.set_index(['timestamps'])

# Feature extraction

barometerFeatures['air_pressure_mean'] = barometerDf['airPressure'].rolling('1s').mean()
barometerFeatures['air_pressure_std'] = barometerDf['airPressure'].rolling('1s').std()
barometerFeatures['air_pressure_var'] = barometerDf['airPressure'].rolling('1s').var()
barometerFeatures['air_pressure_min'] = barometerDf['airPressure'].rolling('1s').min()
barometerFeatures['temperature_max'] = barometerDf['temperature'].rolling('1s').max()
barometerFeatures['temperature_mean'] = barometerDf['temperature'].rolling('1s').mean()
barometerFeatures['temperature_std'] = barometerDf['temperature'].rolling('1s').std()
barometerFeatures['temperature_var'] = barometerDf['temperature'].rolling('1s').var()
barometerFeatures['label'] = barometerDf['label']

# Merge Features

accMagFeatures['timestamps'] = accMagFeatures.index
barometerFeatures['timestamps'] = barometerFeatures.index

accMagFeatures = accMagFeatures.sort_values(by='timestamps')
barometerFeatures = barometerFeatures.sort_values(by='timestamps')

allFeatures = pd.merge_asof(accMagFeatures, barometerFeatures, on='timestamps', by='label',
                            tolerance=pd.Timedelta('100ms'))

# Remove Null Features

print(len(accMagFeatures.index))
print(len(barometerFeatures.index))
print(len(allFeatures.index))

allFeatures = allFeatures[~(allFeatures.temperature_mean.isnull()) & ~(allFeatures.acc_mag_mean.isnull())]

len(allFeatures.index)

# Run Classification For Pressure & Accelerometer

h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(allFeatures)

continuous_feature_columns = [
    'air_pressure_mean',
    'air_pressure_std',
    'air_pressure_var',
    'air_pressure_min',
    'temperature_max',
    'temperature_mean',
    'temperature_std',
    'temperature_var',
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
    model_id="EatingDrinkingElevatorEscalatorAccPresTemp",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
