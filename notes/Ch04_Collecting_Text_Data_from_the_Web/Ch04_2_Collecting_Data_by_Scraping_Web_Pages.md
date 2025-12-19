## Ch04.2 Collecting Data by Scraping Web Pages



### 4.2.1 相关概念

从网页中收集数据并提取信息的过程称为 **网络爬虫（web scraping）**。



### 4.2.2 BeautifulSoup 实战

详见 `Exercise40.ipynb`，利用 `BeautifulSoup` 工具库可以读取 `HTML` 文件内容：

```python
from bs4 import BeautifulSoup

with open('data_ch4/sample_doc.html') as f:
    soup = BeautifulSoup(f, 'html.parser')
    
# 查看纯文本（或标签中间的文字内容）
soup.text
''''\n\n\n A sample HTML Page \n\n\nI am staying at  Mess on No. 72, Banamali Naskar Lane, Kolkata. \nSherlock  stays at 221B, Baker Street, London, UK. \nHamlet said to Horatio,   There are more things in heaven and earth, Horatio,  Than are dreamt of in your philosophy. \n A table denoting details of students\n\n\nname\nqualification\nadditional qualification\nother qualification\n\n\nGangaram\nB.Tech\nNA\nNA\n\n\nGanga\nB.A.\nNA\nNA\n\n\nRam\nB.Tech\nM.Tech\nNA\n\n\nRamlal\nB.Music\nNA\nDiploma in Music\n\n\n\n''''

# 查看列表形式的内容
soup.contents

# 匹配第一个 HTML 标签 address
soup.find('address')

# 匹配所有 HTML 标签 address，返回一个检索列表
soup.find_all('address')

# 查询到的 HTML 节点可继续 find_all
table = soup.find('table')
for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)
```



> **DeepSeek 给出的 BeautifulSoup 常用接口**
>
> 除了 `find()` 和 `find_all()`，BeautifulSoup 还有以下常用方法：
>
> 1. **搜索方法**
>
> ```python
> # CSS选择器
> soup.select('div.content')  # 返回所有匹配元素
> soup.select_one('div.content')  # 返回第一个匹配元素
> 
> # 按属性搜索
> soup.find_all(attrs={"class": "item"})
> ```
>
> 2. **导航方法**
>
> ```python
> # 父节点
> tag.parent  # 直接父节点
> tag.parents  # 所有祖先节点（生成器）
> 
> # 兄弟节点
> tag.next_sibling
> tag.previous_sibling
> tag.next_siblings  # 所有后续兄弟节点
> tag.previous_siblings  # 所有前面兄弟节点
> 
> # 子节点
> tag.contents  # 直接子节点列表
> tag.children  # 直接子节点（生成器）
> tag.descendants  # 所有后代节点（生成器）
> ```
>
> 3. **内容提取**
>
> ```python
> # 文本处理
> tag.get_text()  # 获取所有文本内容
> tag.string  # 如果只有一个子字符串节点
> tag.strings  # 所有字符串生成器
> tag.stripped_strings  # 去除空白的字符串生成器
> 
> # 属性处理
> tag['class']  # 获取属性值
> tag.get('href')  # 安全获取属性
> tag.attrs  # 所有属性字典
> ```
>
> 4. **标签操作**
>
> ```python
> # 标签名
> tag.name  # 标签名称
> tag.has_attr('class')  # 检查是否有某属性
> 
> # 修改
> tag.name = 'new_tag'  # 修改标签名
> tag['class'] = 'new_class'  # 修改属性
> tag.string = 'new text'  # 修改文本
> ```
>
> 5. **其他实用方法**
>
> ```python
> # 检查条件
> tag.has_attr('id')  # 是否有id属性
> tag.get('href', default='')  # 带默认值的属性获取
> 
> # 解包
> tag.decode()  # 转换为字符串
> tag.encode()  # 编码为字节
> tag.prettify()  # 格式化输出
> ```
>
> 
