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

accDfFrames = [accDfCar, accDfBus, accDfLightRail, accDfStanding, accDfWalking, accDfEating, accDfElevator,
               accDfEscalator]

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

# Loading Pressure Data

drivingCarPressureFiles = glob.glob(drivingCarPath + "/0_Pressure*.csv")
drivingCarPressureDf = pd.DataFrame()
list_ = []
for file_ in drivingCarPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarPressureDf = pd.concat(list_)
drivingCarPressureDf['label'] = 'driving car'

busPressureFiles = glob.glob(busPath + "/trim_0_Pressure*.csv")
busPressureDf = pd.DataFrame()
list_ = []
for file_ in busPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
busPressureDf = pd.concat(list_)
busPressureDf['label'] = 'bus'

lightRailPressureFiles = glob.glob(lightRailPath + "/trim_0_Pressure*.csv")
lightRailPressureDf = pd.DataFrame()
list_ = []
for file_ in lightRailPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
lightRailPressureDf = pd.concat(list_)
lightRailPressureDf['label'] = 'light rail'

standingPressureFiles = glob.glob(standingPath + "/trim_0_Pressure*.csv")
standingPressureDf = pd.DataFrame()
list_ = []
for file_ in standingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
standingPressureDf = pd.concat(list_)
standingPressureDf['label'] = 'standing'

walkingPressureFiles = glob.glob(walkingPath + "/trim_0_Pressure*.csv")
walkingPressureDf = pd.DataFrame()
list_ = []
for file_ in walkingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
walkingPressureDf = pd.concat(list_)
walkingPressureDf['label'] = 'walking'

eatingPressureFiles = glob.glob(eatingPath + "/0_Pressure*.csv")
eatingPressureDf = pd.DataFrame()
list_ = []
for file_ in eatingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingPressureDf = pd.concat(list_)
eatingPressureDf['label'] = 'eating'

elevatorPressureFiles = glob.glob(elevatorPath + "/trim_0_Pressure*.csv")
elevatorPressureDf = pd.DataFrame()
list_ = []
for file_ in elevatorPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorPressureDf = pd.concat(list_)
elevatorPressureDf['label'] = 'elevator'

escalatorPressureFiles = glob.glob(escalatorPath + "/trim_0_Pressure*.csv")
escalatorPressureDf = pd.DataFrame()
list_ = []
for file_ in escalatorPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
escalatorPressureDf = pd.concat(list_)
escalatorPressureDf['label'] = 'escalator'

pressureDfFrames = [drivingCarPressureDf, busPressureDf, lightRailPressureDf, standingPressureDf, walkingPressureDf,
                    eatingPressureDf, elevatorPressureDf, escalatorPressureDf]

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
    'pressure_mean',
    'pressure_std',
    'pressure_var',
    'pressure_min',
    'pressure_max',
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
    model_id="AllActivitiesAccelerometerPressure",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
