data = pd.DataFrame({'nums': [10,20,30,10]})
q = MyCorr(data, num_cols = {'nums': {'method': 'absolute', 'num_for_fill': 'mean'}})

1. Нормализация  данных
[10,20,30,10] -> [0.  0.5 1.  0. ]
10-10, (20-10)/(20), (30-10)/(20), 10-10

col = np.array([10,20,30,10])
ms = MinMaxSc()
ms.fit(col)
print(ms.transform(col))


1. Нормализация  данных
[10,10,10, 10] -> [1.  1 1.  1. ]
col = np.array([10,10,10,10])
ms = MinMaxSc()
ms.fit(col)
print(ms.transform(col))

2. Расчет матрицы расстояний
col = [0.  0.5 1.  0. ]
Без учета веса
[[0.  0.5 1.  0. ]
 [0.5 0.  0.5 0.5]
 [1.  0.5 0.  1. ]
 [0.  0.5 1.  0. ]]

3. Добавление значения
15

1. Нормализация  данных
[15] -> [(15-10)/(20)] 0.25
col = [0.  0.5 1.  0. ]
0.25
0.25, 0.25, 0.75, 0.25, 0

1. Нормализация  данных
[15, 25] -> [(15-10)/(20)] 0.25
col = [0.  0.5 1.  0. ]
0.25
0.25, 0.25, 0.75, 0.25, 0