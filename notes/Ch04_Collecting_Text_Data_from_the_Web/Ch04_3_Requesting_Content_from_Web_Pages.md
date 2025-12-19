## Ch04.3 Requesting Content from Web Pages



### 4.3.1 利用 requests 模块

详见 `Exercise41`：

```python
import requests

# Let's read the text version of david copper field available online
r = requests.post('https://www.gutenberg.org/files/766/766-0.txt')

r.status_code

r.text[:1000]

with open("data_ch4/David_Copperfield.txt", 'w', encoding='utf-8') as f:
    f.write(r.text)
```



### 4.3.2 利用 Urllib3 模块

详见 `Exercise41`：

```python
import urllib3
http = urllib3.PoolManager()
rr = http.request('GET', 'https://www.gutenberg.org/files/766/766-0.txt')
rr.status

rr.data[:1000]

with open("data_ch4/David_Copperfield_new.txt", 'wb') as ff:
    ff.write(rr.data)
```



### 4.3.3 抓取经 Jupyter 转化的 HTML 网页

详见 `Exercise42`：

```python
from bs4 import BeautifulSoup

with open('data_ch4/text_classifier.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')
soup.text[:100]

len(soup.find_all('img')) # 3

[i.get_text() for i in soup.find_all('span',attrs={"class":"nn"})]
'''\
['pandas',
 'pd',
 'seaborn',
 'sns',
 'matplotlib.pyplot',
 'plt',
 're',
 'string',
 'nltk',
 'nltk.corpus',
 'nltk.stem',
 'sklearn.feature_extraction.text',
 'sklearn.model_selection',
 'pylab',
 'nltk',
 'warnings',
 'sklearn.metrics',
 'sklearn.linear_model',
 'sklearn.ensemble',
 'xgboost']'''

for md,i in zip(soup.find_all('h2'), soup.find_all('div',attrs={"class":"output_subarea output_stream output_stdout output_text"})):
    print("Model: ",md.get_text())
    print(i.get_text())
    print("---------------------------------------------------------\n\n\n")
```



### 4.3.4 抓取维基百科网页数据

详见 `Activity6`。实测过程中，目标网页结构已经变动，根据样式类定位元素已经无法正常运行。

另外，直接发送 `GET` 请求也返回 `403` 状态码，提示：` 'Please set a user-agent and respect our robot policy https://w.wiki/4wJS. See also https://phabricator.wikimedia.org/T400119.\n'`。即明确要求设置 `User-Agent` 头信息。添加如下头信息恢复正常：

```python
import requests
from bs4 import BeautifulSoup
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
r = requests.get('https://en.wikipedia.org/wiki/Rabindranath_Tagore', headers=headers)
r.status_code, r.text
```



此外，`Activity7` 也存在页面过期的问题，`Packt` 官方常见问题页已从此前的 `https://www.packtpub.com/books/info/packt/faq` 更新为 `https://www.packtpub.com/en-us/help/faqs`，并且问答内容也调整为动态展示，练习参考答案的写法已经全部过时了。

另外，`Activity7` 后半部分演示了正则表达式和 `pandas` 模块在提取电邮信息时的具体应用。不过用到的目标页面——`Packt` 官方的条款与条件页，也从 `https://www.packtpub.com/en-us/help/terms-and-conditions` 更新为 `https://www.packtpub.com/en-us/help/terms-and-conditions`。这部分内容仅供参考，无法作为最新答案。



