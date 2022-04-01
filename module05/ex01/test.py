from TinyStatistician import TinyStatistician

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]

print('list', a)
print('mean', tstat.mean(a))
print('median', tstat.median(a))
print('quartile', tstat.quartile(a))
print('percentile 10', tstat.percentile(a, 10))
print('percentile 28', tstat.percentile(a, 28))
print('percentile 83', tstat.percentile(a, 83))
print('var', tstat.var(a))
print('std', tstat.std(a))

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
