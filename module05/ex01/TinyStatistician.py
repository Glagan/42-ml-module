import math


class TinyStatistician:
    def __init__(self):
        pass

    def mean(self, x: list) -> float:
        if not isinstance(x, list) or len(x) == 0:
            return None
        return sum(x) / len(x)

    def median(self, x: list) -> float:
        if not isinstance(x, list) or len(x) == 0:
            return None
        return self.linear_percentile(x, 50)

    def quartile(self, x: list) -> float:
        if not isinstance(x, list) or len(x) == 0:
            return None
        first = self.linear_percentile(x, 25)
        third = self.linear_percentile(x, 75)
        if first is None or third is None:
            return None
        return [first, third]

    def linear_percentile(self, lst: list, percentile: int) -> float:
        """
        Closest ranks with linear interpolation (C = 1)
        https://en.wikipedia.org/wiki/Percentile
        """
        if not isinstance(lst, list) or len(lst) == 0:
            return None
        if not isinstance(percentile, int):
            return None
        lst = sorted(lst)
        count = len(lst)
        x = (percentile / 100) * (count - 1)
        x_floor = math.floor(x)
        value = x_floor if x_floor >= 0 else 0
        value_next = x_floor + \
            1 if (x_floor + 1) < count - 1 else count - 1
        in_serie = lst[value]
        next_in_serie = lst[value_next]
        frac = x - x_floor
        return in_serie + frac * (next_in_serie - in_serie)

    def percentile(self, lst: list, percentile: int) -> float:
        """
        TODO use another percentile...
        Closest ranks with linear interpolation (C = 1)
        https://en.wikipedia.org/wiki/Percentile
        """
        if not isinstance(lst, list) or len(lst) == 0:
            return None
        if not isinstance(percentile, int):
            return None
        lst = sorted(lst)
        count = len(lst)
        x = (percentile / 100) * (count - 1)
        x_floor = math.floor(x)
        value = x_floor if x_floor >= 0 else 0
        value_next = x_floor + \
            1 if (x_floor + 1) < count - 1 else count - 1
        in_serie = lst[value]
        next_in_serie = lst[value_next]
        frac = x - x_floor
        return in_serie + frac * (next_in_serie - in_serie)

    def var(self, x: list) -> float:
        if not isinstance(x, list) or len(x) == 0:
            return None
        mean = self.mean(x)
        return sum((row - mean) ** 2 for row in x) / len(x)

    def std(self, x: list) -> float:
        if not isinstance(x, list) or len(x) == 0:
            return None
        return math.sqrt(self.var(x))
