## Ch03.5 Saving and Loading Models



在模型构建完成且其性能符合预期后，我们可能需要将其保存以便将来使用。这样可以节省重新构建模型所需的时间。

具体可通过 `joblib` 和 `pickle` 模块将模型保存至硬盘。



实战：`Exercise39.ipynb`

要求：为以下句子创建 `tf-idf` 表示。将此 `tf-idf` 模型保存到磁盘上。从磁盘加载并使用该模型。

'Data Science is an overlap between Arts and Science',
'Generally, Arts graduates are right-brained and Science graduates are left-brained',
'Excelling in both Arts and Science at a time becomes difficult',
'Natural Language Processing is a part of Data Science'

```python
import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'Data Science is an overlap between Arts and Science',
    'Generally, Arts graduates are right-brained and Science graduates are left-brained',
    'Excelling in both Arts and Science at a time becomes difficult',
    'Natural Language Processing is a part of Data Science'
]
tfidf_model = TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())
'''
[[0.40332811 0.25743911 0.         0.25743911 0.         0.
  0.40332811 0.         0.         0.31798852 0.         0.
  0.         0.         0.         0.31798852 0.         0.
  0.         0.         0.40332811 0.         0.         0.
  0.42094668 0.        ]
 [0.         0.159139   0.49864399 0.159139   0.         0.
  0.         0.         0.49864399 0.         0.         0.
  0.24932199 0.49864399 0.         0.         0.         0.24932199
  0.         0.         0.         0.         0.         0.24932199
  0.13010656 0.        ]
 [0.         0.22444946 0.         0.22444946 0.35164346 0.35164346
  0.         0.35164346 0.         0.         0.35164346 0.35164346
  0.         0.         0.35164346 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.18350214 0.35164346]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.30887228 0.         0.
  0.         0.         0.         0.30887228 0.39176533 0.
  0.39176533 0.39176533 0.         0.39176533 0.39176533 0.
  0.2044394  0.        ]]'''
```

按 `joblib` 保存及加载：

```python
dump(tfidf_model, 'tfidf_model.joblib') 
tfidf_model_loaded = load('tfidf_model.joblib')
print(tfidf_model_loaded.fit_transform(corpus).todense())
'''
[[0.40332811 0.25743911 0.         0.25743911 0.         0.
  0.40332811 0.         0.         0.31798852 0.         0.
  0.         0.         0.         0.31798852 0.         0.
  0.         0.         0.40332811 0.         0.         0.
  0.42094668 0.        ]
 [0.         0.159139   0.49864399 0.159139   0.         0.
  0.         0.         0.49864399 0.         0.         0.
  0.24932199 0.49864399 0.         0.         0.         0.24932199
  0.         0.         0.         0.         0.         0.24932199
  0.13010656 0.        ]
 [0.         0.22444946 0.         0.22444946 0.35164346 0.35164346
  0.         0.35164346 0.         0.         0.35164346 0.35164346
  0.         0.         0.35164346 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.18350214 0.35164346]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.30887228 0.         0.
  0.         0.         0.         0.30887228 0.39176533 0.
  0.39176533 0.39176533 0.         0.39176533 0.39176533 0.
  0.2044394  0.        ]]'''
```

按 `pickle` 保存及加载：

```python
#Save the model
with open("tfidf_model.pickle.dat", "wb") as f1:
    pickle.dump(tfidf_model, f1)

#To load the saved model
with open("tfidf_model.pickle.dat", "rb") as f2:
    loaded_model = pickle.load(f2)
    
print(loaded_model.fit_transform(corpus).todense())
'''
[[0.40332811 0.25743911 0.         0.25743911 0.         0.
  0.40332811 0.         0.         0.31798852 0.         0.
  0.         0.         0.         0.31798852 0.         0.
  0.         0.         0.40332811 0.         0.         0.
  0.42094668 0.        ]
 [0.         0.159139   0.49864399 0.159139   0.         0.
  0.         0.         0.49864399 0.         0.         0.
  0.24932199 0.49864399 0.         0.         0.         0.24932199
  0.         0.         0.         0.         0.         0.24932199
  0.13010656 0.        ]
 [0.         0.22444946 0.         0.22444946 0.35164346 0.35164346
  0.         0.35164346 0.         0.         0.35164346 0.35164346
  0.         0.         0.35164346 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.18350214 0.35164346]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.30887228 0.         0.
  0.         0.         0.         0.30887228 0.39176533 0.
  0.39176533 0.39176533 0.         0.39176533 0.39176533 0.
  0.2044394  0.        ]]'''
```

