import numpy as np
import pytest
import pandas as pd

from copy import deepcopy

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.observers import Lag


class TestEmbedding:

    @pytest.fixture(scope="class")
    def library_times(self):
        return pd.DatetimeIndex(pd.date_range(start='2021-01-01', periods=10, freq='D'))

    @pytest.fixture(scope="class")
    def data(self, library_times):
        return pd.DataFrame({'A': list(range(len(library_times))), 'B': list(range(len(library_times)))},
                            index=library_times)

    @pytest.fixture(autouse=True)
    def observers(self):
        return [Lag('A', tau=0), Lag('B', tau=0)]

    # Creating an instance of Embedding with valid DATA, observers, library_times, and compile_block=True should
    # initialize the object with the provided DATA, observers, library times, and a flag to compile the state space
    # embedding.
    def test_valid_initialization(self, data, observers, library_times):
        embedding = Embedding(data, observers, library_times, compile_block=True)

        assert embedding.data.equals(data)
        assert embedding.observers == observers
        assert embedding.library_times.equals(library_times)
        assert embedding.frequency == data.index.freq
        assert embedding.dimension == len(observers)
        assert embedding.distance_tree is not None
        assert embedding.block is not None

    # Compiling the state space embedding by calling the compile() method on an initialized Embedding object should
    # build the state space embedding by applying observer functions to the DATA.
    def test_compile_embedding(self, data, observers, library_times):
        embedding = Embedding(data, observers, library_times)

        assert isinstance(embedding.block, pd.DataFrame)
        assert embedding.block.shape[0] == len(library_times)
        assert embedding.block.shape[1] == len(observers)
        assert embedding.distance_tree is not None

    # Retrieving the embedded state space points for a given set of times by calling the get_points() method on an
    # initialized and compiled Embedding object should return a pandas DataFrame of the embedded state space points
    # for the given set of times.
    def test_get_points(self, data, observers, library_times):

        embedding = Embedding(data, observers, library_times)

        points = embedding.get_points(library_times)

        assert isinstance(points, pd.DataFrame)
        assert points.shape[0] == len(library_times)
        assert points.shape[1] == len(observers)

    #  Creating an instance of Embedding with invalid DATA should raise a TypeError.
    def test_invalid_data_type(self, observers, library_times, data):

        # data should always be indexed by a pd.datetimeindex
        non_datatime_indexed_data = deepcopy(data)
        non_datatime_indexed_data.index = [i for i in range(len(data.index))]

        invalid_data = [
            non_datatime_indexed_data
        ]

        for invalid in invalid_data:
            with pytest.raises(TypeError):
                Embedding(invalid, observers, library_times)

    #  Creating an instance of Embedding with invalid observers should raise a TypeError.
    def test_invalid_observers_type(self, data, library_times):

        invalid_observers = ['invalid_observers']

        for invalid in invalid_observers:
            with pytest.raises(TypeError):
                Embedding(data, invalid, library_times)

    #  Creating an instance of Embedding with invalid library_times should raise a TypeError.
    def test_invalid_library_times_type(self, data, observers):

        invalid_library_times = ['invalid_library_times']

        for invalid in invalid_library_times:
            with pytest.raises(TypeError):
                Embedding(data, observers, invalid)

    # Setting the library times for an initialized Embedding object by calling the set_library() method with valid
    # library_times should update the library times for the state space embedding. I've also added a few ancillary tests
    # here for the compile_block=False option
    def test_set_library_valid_library_times(self, data, observers, library_times):

        embedding = Embedding(data, observers, library_times, compile_block=False)

        new_library_times = pd.DatetimeIndex(['2022-01-04', '2022-01-05', '2022-01-06'])
        embedding.set_library(new_library_times)

        assert embedding.library_times.equals(new_library_times)
        assert embedding.block is None
        assert embedding.distance_tree is None

    # Setting the observers for an initialized Embedding object by calling the set_observers() method with valid
    # observers and compile_block=True should update the observers for the state space embedding and compile the
    # embedding.
    def test_set_observers_valid_observers_compile_block_true(self, data, observers, library_times):

        embedding = Embedding(data, observers, library_times[2:])

        new_observers = [Lag('A', tau=0), Lag('A', tau=-1), Lag('A', tau=-2)]
        embedding.set_observers(new_observers, compile_block=True)

        assert embedding.observers == new_observers
        assert embedding.block is not None
        assert embedding.distance_tree is not None
        assert not embedding.get_points(library_times[2:]).empty

    # Compiling the state space embedding on an initialized Embedding object with no observers should raise an
    # AttributeError.
    def test_compile_no_observers(self, data, observers, library_times):
        empty_observers = []

        with pytest.raises(TypeError):
            Embedding(data, empty_observers, library_times, compile_block=True)

    # Retrieving the embedded state space points for a given set of times on an initialized Embedding object with no
    # observers should return the original data at those times
    def test_retrieve_points_no_observers(self, data, library_times):

        empty_observers = []

        embedding = Embedding(data, empty_observers, library_times, compile_block=False)
        times = pd.DatetimeIndex(['2021-01-01', '2021-01-02'])

        points = embedding.get_points(times)

        assert data.loc[times].equals(points)

    # Retrieving the embedded state space points for a given set of times on an initialized Embedding object with
    # invalid times should raise a KeyError.
    def test_retrieve_points_invalid_times(self, data, observers, library_times):

        embedding = Embedding(data, observers, library_times)
        invalid_times = pd.DatetimeIndex(['2022-01-04', '2022-01-05'])

        with pytest.raises(KeyError):
            embedding.get_points(invalid_times)

    #  Setting the observers for an initialized Embedding object with invalid observers should raise a TypeError.
    def test_set_observers_invalid_observers(self, data, observers, library_times):

        embedding = Embedding(data, observers, library_times)
        invalid_observers = [Lag('C', tau=0), Lag('D', tau=0)]

        with pytest.raises(AttributeError):
            embedding.set_observers(invalid_observers)
