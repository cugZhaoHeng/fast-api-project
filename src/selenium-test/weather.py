# 这是一个最简单的爬虫，从中国天气网上面爬取HTML页面，然后从里面find出需要的元素，不涉及到调接口，主要是解析HTLM即可，除了使用html5lib之外，还可以使用其他的如lxml
from logging import Logger

import requests
from requests.models import Response
from bs4 import BeautifulSoup, ResultSet, PageElement, Tag
# 这里可以 import, 也可以不引入，但是最好是引入，因为下面的 soup = BeautifulSoup(text,'lxml') 是按照字符串来识别解析器的，必须确保当前环境中安装了这个库，所以还是引入，作为说明
import html5lib
import lxml

from utils.logger import create_logger
logger: Logger = create_logger(__name__)

def parse_page(url: str) -> None:
    """ request and parse the page and return a :class:`ResultSet <ResultSet>` object.
    :param url: URL for the new :class:`Request` object.

    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36',
    }
    # 直接获取整个网页
    response: Response = requests.get(url=url, headers=headers)
    # 对返回的HTML强制设置 UTF-8 编码
    response.encoding = "utf-8"
    logger.info(f"response.encode:{response.encoding}")
    logger.info(f"response.text: {response.text}")

    text = response.text
    # 需要用到html5lib解析器，去补全html标签，为什么要补全呢？
    soup = BeautifulSoup(text,'lxml')
    # 使用find函数，定位到class="conMidtab" 的元素
    conMidtab: Tag = soup.find('div',class_='conMidtab')
    # 拿到元素之后，继续获取里面 table 标签
    tables: list[Tag] = conMidtab.find_all('table')
    for table in tables:
        trs = table.find_all('tr')[2:]
        for index,tr in enumerate(trs):
            tds = tr.find_all('td')
            city_td = tds[0]
            if index == 0:
                city_td = tds[1]
            city = list(city_td.stripped_strings)[0]
            temp_td = tds[-2]
            temp = list(temp_td.stripped_strings)[0]
            print({'city':city,'temp':temp})

def main():
    url_list = [
        'http://www.weather.com.cn/textFC/hb.shtml',
        # 'http://www.weather.com.cn/textFC/db.shtml',
        # 'http://www.weather.com.cn/textFC/hd.shtml',
        # 'http://www.weather.com.cn/textFC/hz.shtml',
        # 'http://www.weather.com.cn/textFC/hn.shtml',
        # 'http://www.weather.com.cn/textFC/xb.shtml',
        # 'http://www.weather.com.cn/textFC/xn.shtml',
        # 'http://www.weather.com.cn/textFC/gat.shtml',
    ]
    for url in url_list:
        parse_page(url)

if __name__ == '__main__':
    main()