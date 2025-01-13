## Ch01.3 Text Analytics and NLP



概念：**Text Analytics（文本分析）**，是从文本数据中提取 **有意义的见解** 和 **回答问题** 的方法。这些文本数据不一定是人类语言。

NLP 可以大致分为两种类型：

- 自然语言理解（NLU）：指的是一个具有计算能力的无生命物体 **能够理解人类口语** 的过程。
- 自然语言生成（NLG）：指的是这样一个过程：通过该过程，一个具有计算能力的无生命物体 **能够以人类可以理解的语言表达其思想**。



人类与机器交谈时：

- 机器借助 NLU 过程来理解人类语言
- 机器借助 NLG 过程来生成适当的响应并与人类分享，让人类更容易理解



### Exercise 1: Basic Text Analytics

本练习旨在熟悉 `Python` 的基本语法。详见随书源码 `{REPO_ROOT}/Lesson1/Exercise1.ipynb`。

```python
# sentence = "The quick brown fox jumps over the lazy dog"
# 逆序打印某单词
sentence.split()[2] # 'brown'
sentence.split()[2][::-1] # 'nworb'

# 逆序打印一句话
sentence[::-1]
# 'god yzal eht revo spmuj xof nworb kciuq ehT'

# 列表展开式写法
[words[i] for i in range(len(words)) if i%2 == 0] 
# ['The', 'brown', 'jumps', 'the', 'dog']

# 将句子中的每个单词作逆序处理
print(' '.join([word[::-1] for word in words]))
ehT kciuq nworb xof spmuj revo eht yzal god
```

