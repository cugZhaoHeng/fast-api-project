import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# 存储结果的列表
results = []

# 启动浏览器（推荐使用 webdriver-manager 自动管理驱动）
# 安装：pip install webdriver-manager
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
# 可选：设置无头模式（后台运行）
# options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

try:
    # 1. 打开豆瓣图书首页
    driver.get("https://book.douban.com/")
    driver.maximize_window()
    print("已打开豆瓣图书首页")

    # 2. 找到搜索框并输入关键词（例如：“Python”）
    keyword = input("请输入要搜索的书籍关键词：")  # 可替换为固定词如 "Python"
    search_input = wait.until(EC.presence_of_element_located((By.NAME, "search_text")))
    search_input.clear()
    search_input.send_keys(keyword)
    search_input.send_keys(Keys.RETURN)
    print(f"正在搜索关键词：{keyword}")

    # 等待页面加载出书籍列表
    time.sleep(3)

    # 3. 循环翻页，采集前10页数据
    for page in range(1, 11):
        print(f"正在采集第 {page} 页...")

        try:
            # 等待书籍列表加载完成（等待包含书籍的 ul 标签）
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.subject-list")))

            # 查找当前页所有书籍条目
            books = driver.find_elements(By.CSS_SELECTOR, "ul.subject-list > li.subject-item")

            for book in books:
                try:
                    # 书名
                    title_elem = book.find_element(By.CSS_SELECTOR, "h2 a")
                    title = title_elem.get_attribute("title").strip()  # 使用 title 属性更准确
                    link = title_elem.get_attribute("href")

                    # 简介（作者、出版社、出版时间、价格等）
                    pub_info = book.find_element(By.CSS_SELECTOR, "div.pub").text.strip()

                    # 评分（可能没有）
                    try:
                        rating = book.find_element(By.CSS_SELECTOR, "span.rating_nums").text.strip()
                        if not rating or rating == "暂无评分":
                            rating = None
                    except NoSuchElementException:
                        rating = None

                    # 简介描述（如“本书讲述了...”）
                    try:
                        desc = book.find_element(By.CSS_SELECTOR, "p.abstract").text.strip()
                    except NoSuchElementException:
                        desc = ""

                    # 价格信息在 pub 中，但我们已经获取了 pub_info
                    # 示例：'作者: Eric Matthes | 出版社: 人民邮电出版社 | 出版年: 2016-7 | 页数: 459 | 定价: 79.00元'
                    price = None
                    if "定价:" in pub_info:
                        price = [part.strip() for part in pub_info.split("|") if "定价:" in part][0].replace("定价:", "").strip()

                    # 保存数据
                    results.append({
                        "书名": title,
                        "链接": link,
                        "作者/出版信息": pub_info,
                        "评分": rating,
                        "价格": price,
                        "简介": desc
                    })

                except Exception as e:
                    print(f"解析某本书时出错: {e}")
                    continue

        except TimeoutException:
            print("页面加载超时，可能已到最后一页。")
            break

        # 4. 点击“下一页”
        try:
            next_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "span.next a"))
            )
            if "disabled" in next_button.get_attribute("class"):
                print("下一页不可点击，已到最后一页。")
                break

            next_button.click()
            print("已点击下一页")
            time.sleep(2)  # 给页面加载时间

            # 等待新页面加载（等待 subject-list 更新）
            wait.until(EC.staleness_of(books[0] if books else None))
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.subject-list")))

        except TimeoutException:
            print("找不到下一页按钮，可能已到最后一页。")
            break
        except Exception as e:
            print(f"翻页失败: {e}")
            break

finally:
    # 5. 关闭浏览器
    driver.quit()

# 6. 保存数据到 CSV 文件
if results:
    df = pd.DataFrame(results)
    filename = f"douban_books_{keyword}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 数据采集完成，共 {len(results)} 条记录，已保存至 {filename}")
else:
    print("❌ 未采集到任何数据，请检查关键词或网络连接。")