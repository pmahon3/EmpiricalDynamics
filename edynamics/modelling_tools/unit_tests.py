import data_types
import blocks
import data_loaders
import models
from pandas import Series

lag = data_types.Lag(variable_name='Test', tau=-1)
print(lag)

rng = data_types.ObservationRange(0, 1)
print(rng)

types = data_types.ObservationTypes = {
    1: rng
}
print(types)

pandas_series = Series(index=[i for i in range(10)], data=[i for i in range(10)])
series = data_types.TimeSeries(name='Test', data=pandas_series)
print(series)

new_block = blocks.UnivariateBlock(series=series, lags=[lag])
print(new_block)

new_loader = data_loaders.CompetitionLoader(data_directory='test_directory')
print(new_loader)

new_model = models.UnivariateModel(data_loader=new_loader)
print(new_model)
