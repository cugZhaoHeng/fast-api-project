from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time


# 配置
KEYWORD = "Selenium 教程"
MAX_PAGES = 3  # 爬取页数
DRIVER_PATH = r"D:\temp\chromedriver-win64\chromedriver.exe"

# 启动浏览器（这里用有界面模式，方便观察）
driver = webdriver.Chrome(executable_path=r"D:\temp\chromedriver-win64\chromedriver.exe")
driver.get("https://www.baidu.com")

# 输入关键词
search_box = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "kw"))
)
search_box.clear()
search_box.send_keys(KEYWORD)
search_box.send_keys(Keys.ENTER)

# 保存数据的文件
with open("baidu_results.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["标题", "链接"])  # 表头

    for page in range(MAX_PAGES):
        # 等待搜索结果加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h3.t a"))
        )

        # 获取所有搜索结果
        results = driver.find_elements(By.CSS_SELECTOR, "h3.t a")
        for r in results:
            title = r.text
            link = r.get_attribute("href")
            writer.writerow([title, link])
        print(f"✅ 第 {page+1} 页数据已保存")

        # 找到“下一页”按钮并点击
        try:
            next_btn = driver.find_element(By.LINK_TEXT, "下一页 >")
            next_btn.click()
            time.sleep(2)
        except:
            print("❌ 没有更多页面了")
            break

driver.quit()
print("🎉 爬取完成！结果已保存到 baidu_results.csv")
