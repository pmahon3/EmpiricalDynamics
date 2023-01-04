# class Lag defines a lagged coordinate for a given time series variable

class Lag(object):
    def __init__(self, variable_name: str, tau: int):
        self._variable_name = variable_name
        self._tau = tau
        if tau == 0:
            self._lagged_name = variable_name
        else:
            self._lagged_name = variable_name + '_(t' + str(tau) + ')'

    @property
    def variable_name(self):
        return self._variable_name

    @variable_name.setter
    def variable_name(self, value):
        self._variable_name = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value

    @property
    def lagged_name(self):
        return self._lagged_name

    def __eq__(self, other):
        if isinstance(other, Lag):
            return self._tau == other._tau and self._variable_name == other._variable_name
        elif isinstance(other, str):
            return self._lagged_name == other
        elif isinstance(other, int):
            return self._tau == other
        return False

    def __str__(self):
        return 'Lag(variable_name=' + self._variable_name + ', tau=' + str(self._tau) + ')'

    def __repr__(self):
        return self.__str__()
