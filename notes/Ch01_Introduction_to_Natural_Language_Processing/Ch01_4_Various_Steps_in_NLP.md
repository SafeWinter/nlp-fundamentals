## Ch01.4 Various Steps in NLP

本节介绍各种 NLP 预处理任务，并通过练习分别进行演示。



### 1.4.1 分词 Tokenization

分词是将句子拆分为其组成单词的过程。

按照提取标记（`token`）的数量进行分类：

- **unigram**：单元词、一元组，指文本中单个的词或词组。
- **bigram**：双元词、二元组，指文本中相邻两个词组成的词组。
- **trigram**：三元词、三元组，指文本中相邻的三个词组成的词组。
- **n-gram**：N 元组，指的是任何长度为 `n` 的相邻词组。`n` 可以是任意正整数。
  - 书中解释：**n-gram** refers to a sequence of n items from a given text.（给定文本的 `n` 项序列）

#### 练习2：分词

详见随书源码 `{REPO_ROOT}/Lesson1/Exercise2.ipynb`。

```python
import nltk
words = nltk.word_tokenize("I am reading NLP Fundamentals")
# or
from nltk import word_tokenize
words = word_tokenize("I am reading NLP Fundamentals")

print(words)
# ['I', 'am', 'reading', 'NLP', 'Fundamentals']
```



