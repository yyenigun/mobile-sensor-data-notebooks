import pandas as pd
import math as math
import glob
from h2o import h2o

# Loading Pressure Data for Driving Car, Bus, Light Rail, Standing, Walking

drivingCarPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Driving Car/csv'
drivingCarPressureFiles = glob.glob(drivingCarPath + "/0_Pressure*.csv")
drivingCarPressureDf = pd.DataFrame()
list_ = []
for file_ in drivingCarPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
drivingCarPressureDf = pd.concat(list_)
drivingCarPressureDf['label'] = 'driving car'

busPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Bus*'
busPressureFiles = glob.glob(busPath + "/trim_0_Pressure*.csv")
busPressureDf = pd.DataFrame()
list_ = []
for file_ in busPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
busPressureDf = pd.concat(list_)
busPressureDf['label'] = 'bus'

lightRailPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Light_Rail*'
lightRailPressureFiles = glob.glob(lightRailPath + "/trim_0_Pressure*.csv")
lightRailPressureDf = pd.DataFrame()
list_ = []
for file_ in lightRailPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
lightRailPressureDf = pd.concat(list_)
lightRailPressureDf['label'] = 'light rail'

standingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Standing*'
standingPressureFiles = glob.glob(standingPath + "/trim_0_Pressure*.csv")
standingPressureDf = pd.DataFrame()
list_ = []
for file_ in standingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
standingPressureDf = pd.concat(list_)
standingPressureDf['label'] = 'standing'

walkingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Driving-Transit/Transit/labeled_segments/dir_Walking*'
walkingPressureFiles = glob.glob(walkingPath + "/trim_0_Pressure*.csv")
walkingPressureDf = pd.DataFrame()
list_ = []
for file_ in walkingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
walkingPressureDf = pd.concat(list_)
walkingPressureDf['label'] = 'walking'

eatingPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Eating-Drinking/csv'
eatingPressureFiles = glob.glob(eatingPath + "/0_Pressure*.csv")
eatingPressureDf = pd.DataFrame()
list_ = []
for file_ in eatingPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
eatingPressureDf = pd.concat(list_)
eatingPressureDf['label'] = 'eating'

elevatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Elevator*'
elevatorPressureFiles = glob.glob(elevatorPath + "/trim_0_Pressure*.csv")
elevatorPressureDf = pd.DataFrame()
list_ = []
for file_ in elevatorPressureFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(df)
elevatorPressureDf = pd.concat(list_)
elevatorPressureDf['label'] = 'elevator'

escalatorPath = r'/Users/KadriyeDogan/dev/Sample Dataset/Elevator-Escalator/csv-segments/dir_Escalator*'
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

print(len(pressureDf.index))

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

pressureFeatures['timestamps'] = pressureFeatures.index
pressureFeatures = pressureFeatures.sort_values(by='timestamps')

# Run Classification For Pressure & Accelerometer
h2o.init()
h2o.remove_all()

allFeatures = h2o.H2OFrame(pressureFeatures)

continuous_feature_columns = [
    'pressure_mean',
    'pressure_std',
    'pressure_var',
    'pressure_min',
    'pressure_max'

]

random_forest_model = h2o.H2ORandomForestEstimator(
    model_id="AllActivitiesPressure",
    ntrees=20,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(x=continuous_feature_columns,
                          y='label',
                          training_frame=allFeatures)
