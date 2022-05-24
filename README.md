# Learning Notes for《Natural Language Processing Fundamentals》



## 1. Profiles

![Natural Language Processing Fundamentals](assets/cover.png)

|    **Title**    | **Natural Language Processing Fundamentals** [[buy](https://www.packtpub.com/product/natural-language-processing-fundamentals/9781789954043)] |
| :-------------: | :----------------------------------------------------------: |
|   **Author**    |               **Sohom Ghosh , Dwight Gunning**               |
| **Publication** |                      **Packt, 2019.3**                       |
|    **Pages**    |                           **374**                            |

> **Introduction**
>
> If `NLP` hasn't been your forte, *Natural Language Processing Fundamentals* will make sure you set off to a steady start. This comprehensive guide will show you how to effectively use `Python` libraries and `NLP` concepts to solve various problems.
>
> You'll be introduced to natural language processing and its applications through examples and exercises. This will be followed by an introduction to the initial stages of solving a problem, which includes problem definition, getting text data, and preparing it for modeling. With exposure to concepts like advanced natural language processing algorithms and visualization techniques, you'll learn how to create applications that can extract information from unstructured data and present it as impactful visuals. Although you will continue to learn NLP-based techniques, the focus will gradually shift to developing useful applications. In these sections, you'll understand how to apply `NLP` techniques to answer questions as can be used in chatbots.
>
> By the end of this book, you'll be able to accomplish a varied range of assignments ranging from identifying the most suitable type of `NLP` task for solving a problem to using a tool like `spacy` or `gensim` for performing sentiment analysis. The book will easily equip you with the knowledge you need to build applications that interpret human language.



## 2. Outlines

Status available：:heavy_check_mark: (Completed) | :hourglass_flowing_sand: (Working) | :no_entry: (Not Started) | :orange_book: (Read)

| No.  |                      Chapter Title                       |          Status          |
| :--: | :------------------------------------------------------: | :----------------------: |
| Ch00 |                   [Preface](./Ch00.md)                   |    :heavy_check_mark:    |
| Ch01 | [Introduction to Natural Language Processing](./Ch01.md) | :hourglass_flowing_sand: |
| Ch02 |      [Basic Feature Extraction Methods](./Ch02.md)       |        :no_entry:        |
| Ch03 |        [Developing a Text classifier](./Ch03.md)         |        :no_entry:        |
| Ch04 |      [Collecting Text Data from the Web](./Ch04.md)      |        :no_entry:        |
| Ch05 |               [Topic Modeling](./Ch05.md)                |        :no_entry:        |
| Ch06 |   [Text Summarization and Text Generation](./Ch06.md)    |        :no_entry:        |
| Ch07 |            [Vector Representation](./Ch07.md)            |        :no_entry:        |
| Ch08 |             [Sentiment Analysis](./Ch08.md)              |        :no_entry:        |



Powershell script for generating markdown files in batch:

```powershell
# Create 8 empty markdown files named Ch##.md:
for($i=1; $i -le 8; $i=$i+1){ New-Item -Name "Ch$('{0:d2}' -f $i).md"; }
```

 

