import pandas as pd

df = pd.read_excel('Data/BusData04167.xlsx')
df = df.dropna()

df['Date'] = pd.to_datetime(df['Date'], format='%YY%MM%DD')
df['EstimatedArrival'] = pd.to_timedelta(df['EstimatedArrival'])
df['ActualArrival'] = pd.to_timedelta(df['ActualArrival'])
df['CollectTime'] = pd.to_timedelta(df['CollectTime'])

df['EstimatedArrival'] = df['Date'] + df['EstimatedArrival']
df['ActualArrival'] = df['Date'] + df['ActualArrival']
df['CollectTime'] = df['Date'] + df['CollectTime']

df['ServiceNo'] = df['ServiceNo'].map({
                                        '124':'1',
                                        '145':'2',
                                        '166':'3',
                                        '174':'4',
                                        '174e':'5',
                                        '195':'6',
                                        '195A':'7',
                                        '197':'8',
                                        '32':'9',
                                        '51':'10',
                                        '61':'11',
                                        '63':'12',
                                        '80':'13',
                                        '851':'14',
                                        '851e':'15',
                                        '961':'16',
                                        '961M':'17'
                                    })
df['ServiceNo'] = df['ServiceNo'].astype(int)

df['Order'] = df['Order'].astype(str)
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

df['Type'] = df['Type'].map({'SD': 1,
                              'DD': 2
                              })
df['Type'] = df['Type'].astype(int)

df['Load'] = df['Load'].map({'LSD': 1, 
                             'SEA': 2, 
                             'SDA': 3
                             })
df['Load'] = df['Load'].astype(int)

df['DayOfWeek'] = df['DayOfWeek'].astype(int)

df['VehNum'] = df['VehNum'].apply(lambda x: int(float(x)))
df['VehNum'] = df['VehNum'].astype(int)

df['VehCode'] = df['VehCode'].apply(lambda x: '{:.6f}'.format(x))
df['VehCode'] = df['VehCode'].apply(lambda x: int(float(x)))
df['VehCode'] = df['VehCode'].astype(str)

order = ['CollectTime', 'ServiceNo', 'Order', 'Latitude', 'Longitude', 'Type', 'Load', 'DayOfWeek', 'VehNum', 'VehCode', 'EstimatedArrival', 'ActualArrival']

df = df[order]

df = df[df['ActualArrival'] == df['EstimatedArrival']]

df = df.sort_values(['ActualArrival'])

df.to_excel('Data/preprocessed.xlsx', index=False)