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



### 1.4.2 词性标注 PoS Tagging

`PoS` 即 parts of speech，表示词性。所谓词性标注，就是将句子中的单词按照各自的词性分别进行标注，并最终贴上标签（labeling）的过程。

例如，对 `The sky is blue.` 这句话进行词性标注：

```python
from nltk import pos_tag
pos_tag('The sky is blue.'.split())
# [('The', 'DT'), ('sky', 'NN'), ('is', 'VBZ'), ('blue.', 'JJ')]
```

其中 ——

- `DT` = determiner，即决定词 [^1]；
- `NN` = noun, common, singular or mass，表示名词、单数或不可数名词；
- `VBZ` = verb, present tense, 3rd person singular，表示动词，第三人称单数现在时；
- `JJ` = Adjective，即形容词；



#### 练习3：词性标注练习

```python
from nltk import word_tokenize
from nltk import pos_tag
# ['I', 'am', 'reading', 'NLP', 'Fundamentals']
words = word_tokenize("I am reading NLP Fundamentals")
pos_tag(words)
[('I', 'PRP'), # 人称代词
 ('am', 'VBP'), # 动词，非第三人称单数现在时
 ('reading', 'VBG'), # 动词，动名词或现在分词
 ('NLP', 'NNP'), # 专有名词，单数
 ('Fundamentals', 'NNS')] # 名词，复数
```





#### 1.4.2.1. 常见的词性标记一览表

根据 `ChatGPT` 提供的信息，`nltk` 模块常见的词性标记如下：

| 词性 | 描述                                     | 含义                       |
| :--: | :--------------------------------------- | -------------------------- |
|  CC  | Coordinating conjunction                 | 并列连词                   |
|  CD  | Cardinal number                          | 基数词                     |
|  DT  | Determiner                               | 决定词                     |
|  EX  | Existential there                        | 存在性词                   |
|  FW  | Foreign word                             | 外来词                     |
|  IN  | Preposition or subordinating conjunction | 介词或从属连词             |
|  JJ  | Adjective                                | 形容词                     |
| JJR  | Adjective, comparative                   | 形容词，比较级             |
| JJS  | Adjective, superlative                   | 形容词，最高级             |
|  LS  | List item marker                         | 列表项标记                 |
|  MD  | Modal auxiliary verb                     | 情态助动词                 |
|  NN  | Noun, singular or mass                   | 名词，单数或不可数名词     |
| NNS  | Noun, plural                             | 名词，复数               |
| NNP  | Proper noun, singular                    | 专有名词，单数             |
| NNPS | Proper noun, plural                      | 专有名词，复数             |
| PDT  | Predeterminer                            | 前决定词                   |
| POS  | Possessive ending                        | 所有格结尾                 |
| PRP  | Personal pronoun                         | 人称代词                   |
| PRP$ | Possessive pronoun                       | 所有格代词                 |
|  RB  | Adverb                                   | 副词                       |
| RBR  | Adverb, comparative                      | 副词，比较级               |
| RBS  | Adverb, superlative                      | 副词，最高级               |
|  RP  | Particle                                 | 小品词                     |
|  TO  | "to" (as a preposition)                  | “到”（作为介词）           |
|  UH  | Interjection                             | 感叹词                     |
|  VB  | Verb, base form                          | 动词，基本形式             |
| VBD  | Verb, past tense                         | 动词，过去式               |
| VBG  | Verb, gerund or present participle       | 动词，动名词或现在分词     |
| VBN  | Verb, past participle                    | 动词，过去分词             |
| VBP  | Verb, non-3rd person singular present    | 动词，非第三人称单数现在时 |
| VBZ  | Verb, 3rd person singular present        | 动词，第三人称单数现在时   |
| WDT  | Wh-determiner                            | 疑问决定词                 |
|  WP  | Wh-pronoun                               | 疑问代词                   |
| WP$  | Possessive wh-pronoun                    | 所有格疑问代词             |
| WRB  | Wh-adverb                                | 疑问副词                   |



### 1.4.3. 移除停用词 Stop Word Removal

停用词是用于支持句子构建的常见词汇， 如 `a`、`am` 和 `the`，对句子的意义没有影响。由于经常高频出现，且对句子意义的影响不大，因此需要将其去除。

#### 练习4：停用词

```python
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('English')
print(stop_words)
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

sentence = "I am learning Python. It is one of the most popular programming languages"
sentence_words = word_tokenize(sentence)
print(sentence_words)
# ['I', 'am', 'learning', 'Python', '.', 'It', 'is', 'one', 'of', 'the', 'most', 'popular', 'programming', 'languages']

sentence_no_stops = ' '.join([word for word in sentence_words if word not in stop_words])
print(sentence_no_stops)
# I learning Python . It one popular programming languages
```



### 1.4.4 文本的规范化处理 Text Normalization

文本规范化是一个将文本的不同变体形式转换为标准形式的过程。如 `does` 和 `doing` 处理成 `do`。

常见的处理方法包括：

1. 拼写纠正（spelling correction）
2. 词干提取（stemming）
3. 词形还原（lemmatization）



#### 练习 5：文本规范化

```python
sentence = "I visited US from UK on 22-10-18"
normalized_sentence = sentence.replace("US", "United States").replace("UK", "United Kingdom").replace("-18", "-2018")
print(normalized_sentence)
# I visited United States from United Kingdom on 22-10-2018
```



### 1.4.5 拼写纠正 Spelling Correction

拼写纠正是任何自然语言处理项目中最重要的任务之一。本书采用 `autocorrect` 库来纠正拼写。

实测时发现（2025-1-14），最新的工具函数 `autocorrect.spell(str)` 已经升级为通过类 `Speller` 中的工具方法来实现：

```python
# Deprecated version:
# from autocorrect import spell
# result = spell('Natureal')

# Latest version (2025-1-14):
from autocorrect import Speller
corr = Speller()
spell = corr.autocorrect_word
result = spell('Natureal')
# 'Natural'
```



#### 练习 6：单词和句子的拼写纠正

详见随书源码 `{REPO_ROOT}/Lesson1/Exercise6.ipynb`。

根据源码整理后的 DIY 实测代码：

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from nltk import word_tokenize
from autocorrect import Speller
spell = Speller().autocorrect_word

raw_str = 'Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes'
words = word_tokenize(raw_str)
words
sentence_corrected = ' '.join([spell(word) for word in words])
sentence_corrected
```

运行结果：

```markdown
['Ntural',
 'Luanguage',
 'Processin',
 'deals',
 'with',
 'the',
 'art',
 'of',
 'extracting',
 'insightes',
 'from',
 'Natural',
 'Languaes']
'Natural Language Processing deals with the art of extracting insights from Natural Languages'
```



### 1.4.6 词干提取 Stemming

在英语等语言中，单词在句子中会以各种形式出现，有必要将它们转换为基本形式，这个转换过程就叫词干提取。例如，将 `production` 和 `products` 统一处理成 `product`：

![](../assets/1-1.png)

##### 图 1.1 单词 product 的词干提取示意图



#### 练习 7：词干提取

详见随书源码 `{REPO_ROOT}/Lesson1/Exercise7.ipynb`。

```python
import nltk
stemmer = nltk.stem.PorterStemmer()
stemmer.stem("battling") # 'battl'
```

从上述实测结果可知，英语单词的词干也并非总是名词本身（`battl`）。



### 1.4.7 词形还原 Lemmatization

该过程就是为了解决 `battl` 这样的词干提取问题。

该过程通常会进行额外的检查，并通过查阅字典正确提取出单词的基本形式（额外检查会放缓 NLP 任务的处理进度）。

#### 练习 8：使用词形还原提取基本词汇

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words = ['products', 'production', 'coming', 'battle']
[lemmatizer.lemmatize(x) for x in words]
# ['product', 'production', 'coming', 'battle']
```

书中未作解释，感觉词形还原这步意义不大。

ChatGPT 给出的解释为：

- `products` 被还原为 `product`，是正确的。
- `production` 和 `battle` 没有变化，因为它们的基本形式就是它们本身。
- `coming` 也没有变化，因为它的基本形式是 `come`，但默认情况下，`lemmatizer` 处理时将其视为名词。

改进意见：

```python
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
words = ['products', 'production', 'coming', 'battle']

# 指定词性的词形还原
lemmatized_words = [
    # 这里假设所有单词都是动词
    lemmatizer.lemmatize(x, pos=wordnet.VERB) for x in words
]
print(lemmatized_words)
# ['products', 'production', 'come', 'battle']
```

其中， 参数 `pos` 支持下列合法值：

- `wordnet.NOUN`：名词
- `wordnet.VERB`：动词
- `wordnet.ADJ`：形容词
- `wordnet.ADV`：副词



### 1.4.8 命名实体识别 NER

**NER**，即 **named entity recognition**，命名实体识别。该过程的主要目标在于识别命名实体（如专有名词），并将其对应到定义好的类别中（如人名、地名等等）。

#### 练习 9：处理命名实体

```python
import nltk
from nltk import word_tokenize
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab') # newly added
nltk.download('words')

sentence = "We are reading a book published by Packt which is based out of Birmingham."
i = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=True)
[a for a in i if len(a)==1]
# [Tree('NE', [('Packt', 'NNP')]), Tree('NE', [('Birmingham', 'NNP')])]
```

可以看到，上述代码识别了命名实体 `Packt` 和 `Birmingham`，并将它们映射到一个已定义的类别：`NNP`（即 NNP  Proper noun, singular，表示单数的专有名词）。



### 1.4.9 词义消歧 Word Sense Disambiguation





---

[^1]: **决定词（determiner）** 是一种用来限定名词的词，通常出现在名词前，帮助提供名词的特定信息，如数量、所有权或确定性等。