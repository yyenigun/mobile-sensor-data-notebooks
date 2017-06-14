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
accMagFeatures['start'] = accMagDf['start']
accMagFeatures['label'] = accMagDf['label']

# Loading Pressure Data

drivingCarPressureFiles = glob.glob(drivingCarPath + "/0_Pressure*.csv")
drivingCarPressureDf = pd.DataFrame()
list_ = []
for file_ in drivingCarPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarPressureDf = pd.concat(list_)
drivingCarPressureDf['label'] = 'driving car'

transitPressureFiles = glob.glob(transitPath + "/0_Pressure*.csv")
transitPressureDf = pd.DataFrame()
list_ = []
for file_ in transitPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
transitPressureDf = pd.concat(list_)
transitPressureDf['label'] = 'transit'

pressureDfFrames = [drivingCarPressureDf, transitPressureDf]

pressureDf = pd.concat(pressureDfFrames)

# convert timestamps to datetime
pressureDf['start'] = pd.to_datetime(pressureDf['start'])
pressureDf['end'] = pd.to_datetime(pressureDf['end'])

# Windowing Pressure Data

pressureFeatures = pd.DataFrame()

pressureDf['timestamps'] = pd.to_datetime(pressureDf['timestamps'])
pressureDf.set_index(['timestamps'])
pressureDf = pressureDf.sort_values(by='timestamps')

pressureDfCal = pressureDf[['timestamps', 'start', 'pressure', 'label']]
pressureDfCal = pressureDfCal.set_index(['timestamps'])

pressureFeatures['pressure_mean'] = pressureDfCal['pressure'].rolling('1s').mean()
pressureFeatures['pressure_std'] = pressureDfCal['pressure'].rolling('1s').std()
pressureFeatures['pressure_var'] = pressureDfCal['pressure'].rolling('1s').var()
pressureFeatures['pressure_min'] = pressureDfCal['pressure'].rolling('1s').min()
pressureFeatures['pressure_max'] = pressureDfCal['pressure'].rolling('1s').max()
pressureFeatures['start'] = pressureDfCal['start']
pressureFeatures['label'] = pressureDfCal['label']

# Merge Features

accMagFeatures['timestamps'] = accMagFeatures.index
pressureFeatures['timestamps'] = pressureFeatures.index

accMagFeatures = accMagFeatures.sort_values(by='timestamps')
pressureFeatures = pressureFeatures.sort_values(by='timestamps')

allFeatures = pd.merge_asof(accMagFeatures, pressureFeatures, on='timestamps', by='label',
                            tolerance=pd.Timedelta('100ms'))

# Remove Null Features

print(len(accMagFeatures.index))
print(len(pressureFeatures.index))
print(len(allFeatures.index))

allFeatures = allFeatures[~(allFeatures.pressure_mean.isnull()) & ~(allFeatures.acc_mag_mean.isnull())]

len(allFeatures.index)

# Run Classification For Pressure & Accelerometer
h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(allFeatures)

continuous_feature_columns = [
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
    model_id="ActivityRecognitionModelAccPressure",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
