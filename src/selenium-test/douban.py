import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service as ChromeService
import sys
import os
from utils.logger import create_logger

current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
driver_path = os.path.join(project_root, "data/chromedriver-win64/chromedriver.exe")
logger = create_logger(__name__)
# 存储结果的列表
results = []

# 启动浏览器（推荐使用 webdriver-manager 自动管理驱动）
# 安装：pip install webdriver-manager
from webdriver_manager.chrome import ChromeDriverManager
logger.info("====豆瓣爬虫准备开始====")
options = webdriver.ChromeOptions()
# 可选：设置无头模式（后台运行）
# options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

service = ChromeService(executable_path=driver_path)
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver = webdriver.Chrome(service=service, options=options)
logger.info("driver创建成功")
wait = WebDriverWait(driver, 10)
url = "https://book.douban.com/"

try:
    # 1. 打开豆瓣图书首页
    logger.info(f"尝试访问地址: {url}")
    driver.get(url)
    logger.info("地址访问成功")
    # 窗口最大化是必须的吗？如果不最大化会怎么样？
    driver.maximize_window()
    print("已打开豆瓣图书首页")

    # 2. 找到搜索框并输入关键词（例如：“Python”）
    # 这里需要等待用户输入，那么能不能直接写死呢？
    # keyword = input("请输入要搜索的书籍关键词：")
    keyword = "东野圭吾"
    # 这里要等待 HTML 中的 name=search_text 元素加载出来，才能进行下一步
    logger.info(f"搜索框输入关键词:{keyword}")
    search_input = wait.until(EC.presence_of_element_located((By.NAME, "search_text")))
    # 清空输入框
    search_input.clear()
    # 将预设的关键词塞到输入框中
    search_input.send_keys(keyword)
    # keys.RETURN 是回车键吗？我觉得这里应该是寻找到搜索按钮，然后触发点击事件
    # search_input.send_keys(Keys.RETURN)
    logger.info(f"触发点击事件")
    search_button = driver.find_element(By.CLASS_NAME, "inp-btn")
    search_button.click()
    logger.info(f"正在搜索关键词：{keyword}")

    # 等待页面加载出书籍列表
    # 这里为什么是等待3s，如果等待3s之后也没有出来该怎么办？我觉得这里应该是等待某个标识被加载出来
    time.sleep(3)

    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "item-root")))
    items: list[WebElement] = driver.find_elements(By.CLASS_NAME, "item-root")

    for item in items:
        try:
            detail: WebElement = item.find_element(By.CLASS_NAME, "detail")
            title: WebElement = detail.find_element(By.CLASS_NAME, "title")
            text: str = title.text
            logger.info(f"当前书名：{text}")
            rating: WebElement = detail.find_element(By.CLASS_NAME, "rating")
            rating_nums: WebElement = rating.find_element(By.CLASS_NAME, "rating_nums")
            logger.info(f"评分：{rating_nums.text}")

            abstract: WebElement = detail.find_element(By.CLASS_NAME, "abstract")
            abstract_text:str = abstract.text
            abstract_list: list[str] = abstract_text.split(sep="/")
            abstract_list = [a.strip() for a in abstract_list]
            logger.info(f"描述：{abstract_list}")
            author: str = abstract_list[0]
            translator: str = abstract_list[1]
            publisher: str = abstract_list[2]
            publish_time = abstract_list[3]
            price: str = abstract_list[4]
            logger.info(f"作者：{author}, 翻译人员： {translator}, 出版社：{publisher}, 出版时间：{publish_time}, 定价：{price}")
        except NoSuchElementException:
            logger.error(f"item: {item.text}解析失败")
            continue



    # 3. 循环翻页，采集前10页数据
    # for page in range(1, 11):
    #     print(f"正在采集第 {page} 页...")
    #
    #     try:
    #         # 等待书籍列表加载完成（等待包含书籍的 ul 标签）
    #         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.subject-list")))
    #
    #         # 查找当前页所有书籍条目
    #         books = driver.find_elements(By.CSS_SELECTOR, "ul.subject-list > li.subject-item")
    #
    #         for book in books:
    #             try:
    #                 # 书名
    #                 title_elem = book.find_element(By.CSS_SELECTOR, "h2 a")
    #                 title = title_elem.get_attribute("title").strip()  # 使用 title 属性更准确
    #                 link = title_elem.get_attribute("href")
    #
    #                 # 简介（作者、出版社、出版时间、价格等）
    #                 pub_info = book.find_element(By.CSS_SELECTOR, "div.pub").text.strip()
    #
    #                 # 评分（可能没有）
    #                 try:
    #                     rating = book.find_element(By.CSS_SELECTOR, "span.rating_nums").text.strip()
    #                     if not rating or rating == "暂无评分":
    #                         rating = None
    #                 except NoSuchElementException:
    #                     rating = None
    #
    #                 # 简介描述（如“本书讲述了...”）
    #                 try:
    #                     desc = book.find_element(By.CSS_SELECTOR, "p.abstract").text.strip()
    #                 except NoSuchElementException:
    #                     desc = ""
    #
    #                 # 价格信息在 pub 中，但我们已经获取了 pub_info
    #                 # 示例：'作者: Eric Matthes | 出版社: 人民邮电出版社 | 出版年: 2016-7 | 页数: 459 | 定价: 79.00元'
    #                 price = None
    #                 if "定价:" in pub_info:
    #                     price = [part.strip() for part in pub_info.split("|") if "定价:" in part][0].replace("定价:", "").strip()
    #
    #                 # 保存数据
    #                 results.append({
    #                     "书名": title,
    #                     "链接": link,
    #                     "作者/出版信息": pub_info,
    #                     "评分": rating,
    #                     "价格": price,
    #                     "简介": desc
    #                 })
    #
    #             except Exception as e:
    #                 print(f"解析某本书时出错: {e}")
    #                 continue
    #
    #     except TimeoutException:
    #         print("页面加载超时，可能已到最后一页。")
    #         break
    #
    #     # 4. 点击“下一页”
    #     try:
    #         next_button = wait.until(
    #             EC.element_to_be_clickable((By.CSS_SELECTOR, "span.next a"))
    #         )
    #         if "disabled" in next_button.get_attribute("class"):
    #             print("下一页不可点击，已到最后一页。")
    #             break
    #
    #         next_button.click()
    #         print("已点击下一页")
    #         time.sleep(2)  # 给页面加载时间
    #
    #         # 等待新页面加载（等待 subject-list 更新）
    #         wait.until(EC.staleness_of(books[0] if books else None))
    #         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.subject-list")))
    #
    #     except TimeoutException:
    #         print("找不到下一页按钮，可能已到最后一页。")
    #         break
    #     except Exception as e:
    #         print(f"翻页失败: {e}")
    #         break

finally:
    # 5. 关闭浏览器
    driver.quit()

# 6. 保存数据到 CSV 文件
if results:
    df = pd.DataFrame(results)
    filename = f"douban_books_{keyword}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 数据采集完成，共 {len(results)} 条记录，已保存至 {filename}")
else:
    logger.info("❌ 未采集到任何数据，请检查关键词或网络连接。")