import numpy as np
from TinyStatistician import TinyStatistician

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]

print('# [1, 42, 300, 10, 59]')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
assert tstat.mean(a) == 82.4, "mean failed"
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
assert tstat.median(a) == 42.0, "median failed"
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
assert tstat.quartile(a) == [10.0, 59.0], "quartile failed"
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
assert tstat.percentile(a, 10) == 4.6, "percentile(10) failed"
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
assert tstat.percentile(a, 28) == 13.840000000000003, "percentile(28) failed"
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
assert tstat.percentile(a, 83) == 136.11999999999995, "percentile(83) failed"
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
assert tstat.var(a) == 12279.439999999999, "var failed"
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))
assert tstat.std(a) == 110.81263465868862, "std failed"

a = [3, 7, 42, 45, 48, 784, 894, 412, 78, 2569]
print('\n# [3, 7, 42, 45, 48, 784, 894, 412, 78, 2569]')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))

a = [0, 100]
print('\n# [0, 100]')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))

a = list(range(1, 100))
print('\n# list(range(1, 100))')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))

a = np.linspace(0, 100, 100)
print('\n# np.linspace(0, 100, 100)')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))

a = np.random.rand(42, 42)
print('\n# np.random.rand(42, 42)')
print('list', a)
print('mean\t', tstat.mean(a))
print('np.mean\t', np.mean(a))
print('median\t\t', tstat.median(a))
print('np.median\t', np.median(a))
print('quartile\t', tstat.quartile(a))
print('np.quantile\t', [
    np.quantile(a, 0.25),
    np.quantile(a, 0.75)
])
print('percentile 10\t', tstat.percentile(a, 10))
print('np.percentile 10', np.percentile(a, 10))
print('percentile 28\t', tstat.percentile(a, 28))
print('np.percentile 28', np.percentile(a, 28))
print('percentile 83\t', tstat.percentile(a, 83))
print('np.percentile 83', np.percentile(a, 83))
print('var\t', tstat.var(a))
print('np.var\t', np.var(a))
print('std\t', tstat.std(a))
print('np.std\t', np.std(a))
