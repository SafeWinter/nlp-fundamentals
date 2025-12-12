## Ch02.3 Cleaning Text Data



绝大多数文本数据由于存在各种符号或链接而无法直接使用，需要先进行 **数据清洗**——一门通过剔除原始数据冗余细节，提取有效成分的学问。

例如：

> He tweeted, 'Live coverage of General Elections available at this. tv/show/ge2019. \_/\\\_ Please tune in :) '.
>
> （他发推文说：“大选直播请锁定链接：tv/show/ge2019。:pray: 敬请关注 :grinning: ”。）

常用的处理手法：

- 分词（**Tokenization**）
- 词干提取（**Stemming**）



### 2.3.1 演示分词用到的三种方法

- `Python` 的正则表达式库 `re`；
- `nltk` 库中的 `ngrams` 函数；
- `TextBlob` 库中的 `ngrams` 函数；

要处理的原始文本内容为：

```python
sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'
```

正则库 `re` 的方式：

```python
import re
re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
```

为保留上下文信息，需要提取多元词组（自定义处理函数）：

```python
import re
def n_gram_extractor(sentence, n):
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])

n_gram_extractor('The cute little boy is playing with the kitten.', 3)
'''['The', 'cute', 'little']
['cute', 'little', 'boy']
['little', 'boy', 'is']
['boy', 'is', 'playing']
['is', 'playing', 'with']
['playing', 'with', 'the']
['with', 'the', 'kitten']'''
```

`nltk` 的方式：

```python
from nltk import ngrams
list(ngrams('The cute little boy is playing with the kitten.'.split(), 3))
'''[('The', 'cute', 'little'),
 ('cute', 'little', 'boy'),
 ('little', 'boy', 'is'),
 ('boy', 'is', 'playing'),
 ('is', 'playing', 'with'),
 ('playing', 'with', 'the'),
 ('with', 'the', 'kitten.')]'''
```

`TextBlob` 的方式：

```python
from textblob import TextBlob
blob = TextBlob("The cute little boy is playing with the kitten.")
blob.ngrams(n=3)
'''[WordList(['The', 'cute', 'little']),
 WordList(['cute', 'little', 'boy']),
 WordList(['little', 'boy', 'is']),
 WordList(['boy', 'is', 'playing']),
 WordList(['is', 'playing', 'with']),
 WordList(['playing', 'with', 'the']),
 WordList(['with', 'the', 'kitten'])]'''
```

注意：引入 `textblob` 模块前需要先在 `Anaconda` 对应环境中安装该模块：`conda install -c conda-forge textblob`



### 2.3.2 Keras 与 TextBlob 的分词方案对比

`Keras` 和 `TextBlob` 都是用于执行各类 `NLP` 任务的两大主流 `Python` 库，实测时 `Keras` 已经并入 `TensorFlow`，因此模块安装命令需更新为：

```cmd
conda install -c conda-forge tensorflow
```

`TextBlob` 提供了一个简单易用的操作接口，而 `Keras` 则主要用于执行基于深度学习的 `NLP` 任务。

`TextBlob` 分词实现：

```python
#i) Tokenization with TextBlob
from textblob import TextBlob

blob = TextBlob(sentence)
blob.words
 # ['sunil', 'tweeted', 'witnessing', '70th', 'republic', 'day', 'of', 'india', 'from', 'rajpath', 'new', 'delhi', 'mesmerizing', 'performance', 'by', 'indian', 'army', 'awesome', 'airshow', 'india', 'official', 'indian', 'army', 'india', '70threpublic', 'day', 'for', 'more', 'photos', 'ping', 'me', 'sunil', 'photoking', 'com']
```

`Keras` 分词实现：

```python
#ii) Tolenization with Keras
# from keras.preprocessing.text import text_to_word_sequence # 已过时
from tensorflow.keras.preprocessing.text import text_to_word_sequence

text_to_word_sequence(sentence)
# WordList(['Sunil', 'tweeted', 'Witnessing', '70th', 'Republic', 'Day', 'of', 'India', 'from', 'Rajpath', 'New', 'Delhi', 'Mesmerizing', 'performance', 'by', 'Indian', 'Army', 'Awesome', 'airshow', 'india_official', 'indian_army', 'India', '70thRepublic_Day', 'For', 'more', 'photos', 'ping', 'me', 'sunil', 'photoking.com'])
```



### 2.3.3 分词器的类型

主要列举了五种不同的分词器：

1. 推文分词器（**Tweet tokenizer**）：专为推特发文设计的分词工具，能处理推特平台广泛使用的情感表达与情绪表述
2. `MWE` 分词器（**MWE tokenizer**）：`MWE` 即 **Multi-Word Expression** 多词表达式，可将特定的多词组合视为独立单元处理（如 “United States of America”、“People’s Republic of China”、“not only”、“but also”）
3. 正则分词器（**Regular expression tokenizer**）：基于正则表达式开发，按特定模式进行分割
4. 空白符分词器（**Whitespace tokenizer**）：在遇到空格、制表符或换行符时分割字符串
5. 单词标点分词器（**Word Punkt tokenizer**）：可将文本拆分为字母字符、数字和非字母字符的列表

要处理的原始文本为：

```python
sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)"'
```

【示例1】推文分词器

```python
#i) Tweet tokenizer
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(sentence)
```

【示例2】MWE 分词器

```python
#ii) MWE Tokenizer
from nltk.tokenize import MWETokenizer

mwe_tokenizer = MWETokenizer([('Republic', 'Day')]) # Declaring set of words that are to be treated as one entity
mwe_tokenizer.add_mwe(('Indian', 'Army')) # Adding more words to the set
# ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic_Day', 'of', 'India', 'from', 'Rajpath,', 'New', 'Delhi.', 'Mesmerizing', 'performance', 'by', 'Indian', 'Army!', 'Awesome', 'airshow!', '@india_official', '@indian_army', '#India', '#70thRepublic_Day.', 'For', 'more', 'photos', 'ping', 'me', 'sunil@photoking.com', ':)"']
```

由于受感叹号 `!` 的影响，`Indian Army` 并未识别成单个词元，除非移除感叹号：

```python
mwe_tokenizer.tokenize(sentence.replace('!','').split())
# ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic_Day', 'of', 'India', 'from', 'Rajpath,', 'New', 'Delhi.', 'Mesmerizing', 'performance', 'by', 'Indian_Army', 'Awesome', 'airshow', '@india_official', '@indian_army', '#India', '#70thRepublic_Day.', 'For', 'more', 'photos', 'ping', 'me', 'sunil@photoking.com', ':)"']
```

【示例3】正则分词器

```python
#iii) Regexp Tokenizer
from nltk.tokenize import RegexpTokenizer

reg_tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence)
```

【示例4】空白符分词器

```python
#iv) Whitespace Tokenizer
from nltk.tokenize import WhitespaceTokenizer

wh_tokenizer = WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence)
# ['Sunil', 'tweeted,', '"Witnessing', '70th', 'Republic', 'Day', 'of', 'India', 'from', 'Rajpath,', 'New', 'Delhi.', 'Mesmerizing', 'performance', 'by', 'Indian', 'Army!', 'Awesome', 'airshow!', '@india_official', '@indian_army', '#India', '#70thRepublic_Day.', 'For', 'more', 'photos', 'ping', 'me', 'sunil@photoking.com', ':)"']
```

【示例5】单词标点分词器

```python
#v) WordPunct Tokenizer
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer = WordPunctTokenizer()
wp_tokenizer.tokenize(sentence)
# ['Sunil', 'tweeted', ',', '"', 'Witnessing', '70th', 'Republic', 'Day', 'of', 'India', 'from', 'Rajpath', ',', 'New', 'Delhi', '.', 'Mesmerizing', 'performance', 'by', 'Indian', 'Army', '!', 'Awesome', 'airshow', '!', '@', 'india_official', '@', 'indian_army', '#', 'India', '#', '70thRepublic_Day', '.', 'For', 'more', 'photos', 'ping', 'me', 'sunil', '@', 'photoking', '.', 'com', ':)"']
```



### 2.3.4 分词存在的问题

主要是空白字符和连字符可能带来歧义。另外，像中文和日文这样的句子并不以空白符来分隔字词，这类分词工作就很困难。



### 2.3.5 词干提取 Stemming

在英语等语言中，词语在句子中的使用形式会发生变化。将词语恢复为原始形式的过程称为 **词干提取（stemming）**。

本节介绍两种方法：

- `RegexpStemmer` 正则词干提取器
- `Porter stemmer` 波特词干提取器

正则词干提取器用法演示：

```python
# Exercise16.ipynb
from nltk.stem import RegexpStemmer

sentence = "I love playing football"
regex_stemmer = RegexpStemmer('ing$', min=4) # min 参数限定被处理文本的最小长度
' '.join([regex_stemmer.stem(wd) for wd in sentence.split()])
# 'I love play football'
```

波特词干提取器用法演示：

```python
from nltk.stem.porter import PorterStemmer

sentence = "Before eating, it would be nice to sanitize your hands with a sanitizer"
ps_stemmer = PorterStemmer()
' '.join([ps_stemmer.stem(wd) for wd in sentence.split()])
# 'befor eating, it would be nice to sanit your hand with a sanit'
```



### 2.3.6 词形还原 Lemmatization

这一步其实是对词干提取结果的必要修正，防止词干提取后的词元失去实际意义，如 `independence` 变为 `independ`。

词形还原技术通过运用词汇库和词形分析来解决这类问题，它能返回字典中实际存在的基础词形。

具体用法演示如下：

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

lemmatizer = WordNetLemmatizer()

sentence = "The products produced by the process today are far better than what it produces generally."

' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)])
# 'The product produced by the process today are far better than what it produce generally .'
```



> **TextBlob 在单复数词形变化中的应用**
>
> `TextBlob` 提供了出色的单词单复数转换功能：
>
> ```python
> from textblob import TextBlob
> 
> sentence = TextBlob('She sells seashells on the seashore')
> sentence.words
> # WordList(['She', 'sells', 'seashells', 'on', 'the', 'seashore'])
> sentence.words[2].singularize()
> # 'seashell'
> sentence.words[5].pluralize()
> # 'seashores'
> ```



### 2.3.7 语言翻译

不同语言常混合使用以表达内容。此时将整段文本翻译为单一语言就成为分析前不可或缺的预处理步骤：

```python
from textblob import TextBlob

en_blob = TextBlob(u'muy bien')
en_blob.translate(from_lang='es', to='en') 
# TextBlob("very good")
```

不同语言及其语言编码情况如下（实测时发现 [原链接](https://ctrlq.org/code/19899-google-translate-languages) 已失效，最新编码情况详见 [GitHub Gist](https://gist.github.com/sthobis/c52644d079051bc6456c4b20668db813)）：

```js
const languages = {
  'af': 'Afrikaans',
  'sq': 'Albanian',
  'ar': 'Arabic',
  'az': 'Azerbaijani',
  'eu': 'Basque',
  'bn': 'Bengali',
  'be': 'Belarusian',
  'bg': 'Bulgarian',
  'ca': 'Catalan',
  'zh-CN': 'Chinese Simplified',
  'zh-TW': 'Chinese Traditional',
  'hr': 'Croatian',
  'cs': 'Czech',
  'da': 'Danish',
  'nl': 'Dutch',
  'en': 'English',
  'eo': 'Esperanto',
  'et': 'Estonian',
  'tl': 'Filipino',
  'fi': 'Finnish',
  'fr': 'French',
  'gl': 'Galician',
  'ka': 'Georgian',
  'de': 'German',
  'el': 'Greek',
  'gu': 'Gujarati',
  'ht': 'Haitian Creole',
  'iw': 'Hebrew',
  'hi': 'Hindi',
  'hu': 'Hungarian',
  'is': 'Icelandic',
  'id': 'Indonesian',
  'ga': 'Irish',
  'it': 'Italian',
  'ja': 'Japanese',
  'kn': 'Kannada',
  'ko': 'Korean',
  'la': 'Latin',
  'lv': 'Latvian',
  'lt': 'Lithuanian',
  'mk': 'Macedonian',
  'ms': 'Malay',
  'mt': 'Maltese',
  'no': 'Norwegian',
  'fa': 'Persian',
  'pl': 'Polish',
  'pt': 'Portuguese',
  'ro': 'Romanian',
  'ru': 'Russian',
  'sr': 'Serbian',
  'sk': 'Slovak ',
  'sl': 'Slovenian',
  'es': 'Spanish',
  'sw': 'Swahili',
  'sv': 'Swedish',
  'ta': 'Tamil',
  'te': 'Telugu ',
  'th': 'Thai',
  'tr': 'Turkish',
  'uk': 'Ukrainian',
  'ur': 'Urdu',
  'vi': 'Vietnamese',
  'cy': 'Welsh',
  'yi': 'Yiddish'
};

export default languages;
```



### 2.3.8 停用词的移除

像 `am`、`the`、`are` 之类的停用词主要起辅助作用：帮助构建句子结构，但不会改变所在句子的核心含义。因此可以安全地忽略这些词。

```python
from nltk import word_tokenize

sentence = "She sells seashells on the seashore"
custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']
' '.join([word for word in word_tokenize(sentence) if word.lower() not in custom_stop_word_list])
# 'sells seashells seashore'
```

