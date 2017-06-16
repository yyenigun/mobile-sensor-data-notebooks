import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Accelerometer Data for Driving Car, Bus, Light Rail, Standing, Walking

drivingCarPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Driving Car/csv'
drivingCarFiles = glob.glob(drivingCarPath + "/0_Accelerometer*.csv")
accDfCar = pd.DataFrame()
list_ = []
for file_ in drivingCarFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfCar = pd.concat(list_)
accDfCar['label'] = 'driving car'

busPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Bus*'
busFiles = glob.glob(busPath + "/trim_0_Accelerometer*.csv")
accDfBus = pd.DataFrame()
list_ = []
for file_ in busFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfBus = pd.concat(list_)
accDfBus['label'] = 'bus'

lightRailPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Light_Rail*'
lightRailFiles = glob.glob(lightRailPath + "/trim_0_Accelerometer*.csv")
accDfLightRail = pd.DataFrame()
list_ = []
for file_ in lightRailFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfLightRail = pd.concat(list_)
accDfLightRail['label'] = 'light rail'

standingPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Standing*'
standingFiles = glob.glob(standingPath + "/trim_0_Accelerometer*.csv")
accDfStanding = pd.DataFrame()
list_ = []
for file_ in standingFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfStanding = pd.concat(list_)
accDfStanding['label'] = 'standing'

walkingPath = r'/Users/yalcin.yenigun/dev/workspaces/gsu/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Walking*'
walkingFiles = glob.glob(walkingPath + "/trim_0_Accelerometer*.csv")
accDfWalking = pd.DataFrame()
list_ = []
for file_ in walkingFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
accDfWalking = pd.concat(list_)
accDfWalking['label'] = 'walking'

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

accDfFrames = [accDfCar, accDfBus, accDfLightRail, accDfStanding, accDfWalking, accDfEating, accDfElevator, accDfEscalator]

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

# Loading Magnetometer Data

drivingCarMagFiles = glob.glob(drivingCarPath + "/0_Magnetometer*.csv")
drivingCarMagDf = pd.DataFrame()
list_ = []
for file_ in drivingCarMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarMagDf = pd.concat(list_)
drivingCarMagDf['label'] = 'driving car'

busMagFiles = glob.glob(busPath + "/trim_0_Magnetometer*.csv")
busMagDf = pd.DataFrame()
list_ = []
for file_ in busMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
busMagDf = pd.concat(list_)
busMagDf['label'] = 'bus'

lightRailMagFiles = glob.glob(lightRailPath + "/trim_0_Magnetometer*.csv")
lightRailMagDf = pd.DataFrame()
list_ = []
for file_ in lightRailMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
lightRailMagDf = pd.concat(list_)
lightRailMagDf['label'] = 'light rail'

standingMagFiles = glob.glob(standingPath + "/trim_0_Magnetometer*.csv")
standingMagDf = pd.DataFrame()
list_ = []
for file_ in standingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
standingMagDf = pd.concat(list_)
standingMagDf['label'] = 'standing'

walkingMagFiles = glob.glob(walkingPath + "/trim_0_Magnetometer*.csv")
walkingMagDf = pd.DataFrame()
list_ = []
for file_ in walkingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
walkingMagDf = pd.concat(list_)
walkingMagDf['label'] = 'walking'

eatingMagFiles = glob.glob(eatingPath + "/0_Magnetometer*.csv")
eatingMagDf = pd.DataFrame()
list_ = []
for file_ in eatingMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingMagDf = pd.concat(list_)
eatingMagDf['label'] = 'eating'

elevatorMagFiles = glob.glob(elevatorPath + "/trim_0_Magnetometer*.csv")
elevatorMagDf = pd.DataFrame()
list_ = []
for file_ in elevatorMagFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorMagDf = pd.concat(list_)
elevatorMagDf['label'] = 'elevator'

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
    model_id="AllActivitiesAccelerometerMagnetometer",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
