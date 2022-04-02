from TinyStatistician import TinyStatistician

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]

print('list', a)
print('mean', tstat.mean(a))
assert tstat.mean(a) == 82.4, "mean failed"
print('median', tstat.median(a))
assert tstat.median(a) == 42.0, "median failed"
print('quartile', tstat.quartile(a))
assert tstat.quartile(a) == [10.0, 59.0], "quartile failed"
print('percentile 10', tstat.percentile(a, 10))
assert tstat.percentile(a, 10) == 1.0, "percentile(10) failed"
print('percentile 28', tstat.percentile(a, 28))
assert tstat.percentile(a, 28) == 10.0, "percentile(28) failed"
print('percentile 83', tstat.percentile(a, 83))
assert tstat.percentile(a, 83) == 300, "percentile(83) failed"
print('var', tstat.var(a))
assert tstat.var(a) == 12279.439999999999, "var failed"
print('std', tstat.std(a))
assert tstat.std(a) == 110.81263465868862, "std failed"

a = [0, 100]
print('\nlist', a)
print('mean', tstat.mean(a))
print('median', tstat.median(a))
print('quartile', tstat.quartile(a))
print('percentile 10', tstat.percentile(a, 10))
print('percentile 28', tstat.percentile(a, 28))
print('percentile 83', tstat.percentile(a, 83))
print('var', tstat.var(a))
print('std', tstat.std(a))

a = list(range(1, 100))
print('\nlist', a)
print('mean', tstat.mean(a))
print('median', tstat.median(a))
print('quartile', tstat.quartile(a))
print('percentile 10', tstat.percentile(a, 10))
print('percentile 28', tstat.percentile(a, 28))
print('percentile 83', tstat.percentile(a, 83))
print('var', tstat.var(a))
print('std', tstat.std(a))
