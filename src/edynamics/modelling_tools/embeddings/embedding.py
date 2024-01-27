import logging
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from edynamics.modelling_tools.observers import Observer


class Embedding:
    """
    --------------------------------------------------------------------------------------------------------------------
    SUMMARY
    --------------------------------------------------------------------------------------------------------------------
    The Embedding class is used to create a state space embedding by applying observer functions to a time series DATA.
    It allows for the compilation of the state space embedding and retrieval of embedded points based on a given set of
    times. It also provides functionality to set the library times and observers, as well as finding the k nearest
    neighbors of an embedded point.

    --------------------------------------------------------------------------------------------------------------------
    EXAMPLE USAGE
    --------------------------------------------------------------------------------------------------------------------
    >>> data: pd.DataFrame
    >>> observers: List[Observer]
    >>> library_times: pd.DatetimeIndex

    # Create an instance of the Embedding class:

    >>> embedding = Embedding(data, observers, library_times)

    # Compile the state space embedding:

    >>> embedding.compile()

    # Get the embedded points for a given set of times:

    >>> points = embedding.get_points(times)

    # Set the library times:

    >>> embedding.set_library(library_times)

    # Set the observers and compile the state space embedding:

    >>> embedding.set_observers(observers, compile_block=True)

    --------------------------------------------------------------------------------------------------------------------
    MAIN FUNCTIONALITIES
    --------------------------------------------------------------------------------------------------------------------


        -   Create a state space embedding by applying observer functions to a time series data
        -   Compile the state space embedding by building the embedding block and KDTree
        -   Retrieve the embedded state space points for a given set of times
        -   Set the library times and observers
        -   Find the k nearest neighbors of an embedded point

    --------------------------------------------------------------------------------------------------------------------
    METHODS
    --------------------------------------------------------------------------------------------------------------------

    __init__(self, DATA: pd.DataFrame, observers: List[Observer], library_times: pd.DatetimeIndex, compile_block: bool)
        Initializes the Embedding object with the provided DATA, observers, library times, and a flag to compile the
        state space embedding.\n

    compile(self) -> None:
        Builds the state space embedding by applying observer functions to the data.\n

    get_points(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        Retrieves the embedded state space points for a given set of times.\n

    set_library(self, library_times: pd.DatetimeIndex) -> None:
        Sets the library times for the state space embedding.\n

    set_observers(self, observers: List[Observer], compile_block: bool = True) -> None:
        Sets the observers for the state space embedding and optionally compiles the embedding.\n

    get_k_nearest_neighbours(self, point: np.array, max_time: pd.Timestamp, knn: int) -> List[int]:
        Returns the k nearest neighbors of an embedded point within a given maximum time.\n

    --------------------------------------------------------------------------------------------------------------------
    FIELDS
    --------------------------------------------------------------------------------------------------------------------

        data: pd.DataFrame:
            The time series data to be embedded.\n
        observers: List[Observer]:
            The observation functions for the embedding.\n
        library_times: pd.DatetimeIndex:
            The library times for the state space embedding.\n
        frequency: pd.DatetimeIndex.freq:
            The frequency spacing of the time series.\n
        dimension: int:
            The dimension of the embedding, equal to the number of observation
            functions.\n
        distance_tree: cKDTree:
            A KDTree storing the distances between all pairs of library points for the
            delay embedding.\n
        block: pd.DataFrame:
            The pandas dataframe of the delay embedding.

    """

    def __init__(
            self, data: pd.DataFrame, observers: List[Observer], library_times: pd.DatetimeIndex,
            compile_block: bool = True
    ) -> None:
        """
        Defines a state space embedding by applying generic observer functions to a uni- or multivariate time series.

        :param pd.DataFrame data:   Data to be embedded.
        :param observers:           Observation functions for the embedding.
        :param bool compile_block:  If true, compile the state space embedding by applying observer functions to the
                                    DATA.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be of type pd.DataFrame. Got {type(data)}")
        if not isinstance(observers, list) or \
                not all(isinstance(observer, Observer) for observer in observers) or \
                (not observers and compile_block):
            raise TypeError(
                f"observers must be a list of Observer objects or compile_block must be false: Got {type(observers)} and {compile_block}")
        if not isinstance(library_times, pd.DatetimeIndex):
            raise TypeError(f"library_times must be of type pd.DatetimeIndex. Got {type(library_times)}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(f"data index must be a pd.DatetimeIndex. Got {type(data.index)}")

        self.data = data
        self.observers: List[Observer] = observers
        self.library_times = library_times

        #: pd.DatetimeIndex.Freq the frequency spacing of the time series
        self.frequency = data.index.freq
        #: int: dimension of the Embedding, equal to the length of the list of observation functions.
        self.dimension: int = len(observers)
        #: scipy.spatial.cKDTree: a KDTree storing the distances between all pairs of library points for the delay
        # Embedding using the l2 Norm in R^n where n is the Embedding dimension (i.e. number of lags, len(self.lags))
        self.distance_tree: cKDTree = None
        #: pd.DataFrame: pandas dataframe of the delay Embedding
        self.block: pd.DataFrame = None

        if compile_block:
            self.compile()

        logging.info("Embedding created.")

    # PUBLIC
    def compile(self) -> None:
        """
        Builds the state space embedding according to the observations functions.

        :rtype: None
        """

        logging.info(f"Compiling the state space embedding with {len(self.observers)} observers...")
        logging.info(f"Data size: {self.data.shape}, Library times: {len(self.library_times)}")

        if len(set(self.observers)) < len(self.observers):
            raise Warning(f"Duplicate observers assigned to the embedding:\n\t{', '.join(map(str, self.observers))}\n\t"
                          f"This may result in unexpected behaviour.")

        self.block = pd.DataFrame(
            columns=[obs.observation_name for obs in self.observers],
            index=self.library_times,
        )

        # build the embedding block
        # try to keep observation functions vectorized

        for obs in self.observers:
            self.block[obs.observation_name] = obs.observe(data=self.data, times=self.library_times).values

        self.block.dropna(inplace=True)

        # build the KDTree
        self.distance_tree = cKDTree(self.block.iloc[:-1])

        # set dimensions
        self.dimension = len(self.observers)

        logging.info(f"State space embedding compiled. Block size: {self.block.shape}")

    def get_points(self, times: Union[pd.DatetimeIndex, pd.Index]) -> pd.DataFrame:
        """
        Retrieves the embedded state space points for a given set of times.

        :param pd.DatetimeIndex times: the index of times for the desired points
        :return: A dataframe of the embedded points at the given times
        :rtype: pd.DataFrame
        """

        logging.debug("Getting embedded point at times: %s", times)

        if self.observers is None or not self.observers:
            return self.data.loc[times]

        points = pd.DataFrame(index=times, columns=[obs.observation_name for obs in self.observers])

        for obs in self.observers:
            points[obs.observation_name] = obs.observe(data=self.data, times=times).values

        return points

    def set_library(self, library_times: pd.DatetimeIndex) -> None:
        """

        :param library_times:
        :rtype: None
        """
        self.library_times = library_times

    def set_observers(self, observers: List[Observer], compile_block: bool = False) -> None:
        """

        :param observers:
        :param compile_block:
        :rtype: None
        """
        for obs in observers:
            if obs.variable_name not in self.data.columns:
                raise AttributeError(f"The observer {obs} is not valid for the data. {obs.variable_name}"
                                     f" not found in {self.data.columns}")

        logging.info(msg="Setting observers...")
        self.observers = observers

        if compile_block:
            logging.info(msg="Compiling...")
            self.compile()

    def get_k_nearest_neighbours(self,
                                 point: np.array, knn: int) -> List[int]:
        """
        Returns the k nearest neighbours of the Embedding and their distances to the given embedded point.

        :param np.array point: the point for which we want the k nearest neighbours. The point should be a vector, i.e.
        a 1-D np.array.
        :param int knn: the number of nearest neighbours to return.
        :return: a list of the integer indices of the k nearest neighbours in the library block.
        :rtype: List[Int]
        """

        logging.debug("Getting k nearest neighbours of: %s", point)
        dists, indices = self.distance_tree.query(point, k=knn)
        return indices

    def get_ball_point(self,
                       point: np.array,
                       radius: float,
                       p: float = 2) -> List[bool]:
        """
        Returns indices of the Embedding within a radius of a ball centered at the given point.

        :param np.array point: The point in the embedding space for which to find the ball point.
        :param float radius: The radius of the ball in Euclidean space.
        :param float p: The order of the Minkowski distance metric to be used. Defaults to 2, which corresponds to
            Euclidean distance.
        :return: A list of boolean values indicating whether each point in the embedding space falls within the
            specified ball.
        """
        logging.debug("Getting ball point of: %s", point)
        if not isinstance(point, np.ndarray):
            raise TypeError("point must be a numpy array")

        if point.ndim != 1:
            raise ValueError("point must be a 1-dimensional array")
        if len(point) != self.dimension:
            raise ValueError("point must have the same dimension as the embedding")

        indices = self.distance_tree.query_ball_point(x=point,
                                                      r=radius,
                                                      p=p)

        return indices

    def __str__(self):
        return f"Embedding object:\n" \
               f"\tVariables:\t\t{', '.join(map(str, self.data.columns))}\n" \
               f"\tObservers:\t\t{', '.join(map(str, self.observers))}\n" \
               f"\tFrequency:\t\t{self.frequency}\n" \
               f"\tData Length:\t{len(self.data)}\n" \
               f"\tBlock Length:\t{len(self.block)}\n" \
               f"\tData:\n\t\tStart:\t{self.data.index[0]}\n\t\tEnd:\t{self.data.index[-1]}\n" \
               f"\tLibrary:\n\t\tStart:\t{self.library_times[0]}\n\t\tEnd:\t{self.library_times[-1]}"

    def __repr__(self):
        return f"Embedding:\n" \
               f"\tObservers:\t{', '.join(map(repr, self.observers))}\n" \
               f"\tData:\t\t{', '.join(map(repr, self.data.columns))}\n" \
               f"\tFrequency:\t{self.frequency}"

    def __eq__(self, other):
        if isinstance(other, Embedding):
            return self.data.equals(other.data) and self.observers == other.observers and \
                self.library_times.equals(other.library_times)
        return False

    def __hash__(self):
        return hash((self.data, self.observers, self.library_times))

    def __len__(self):
        return self.dimension
