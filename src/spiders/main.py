import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService

service = ChromeService(executable_path=r"D:\software\chromedriver-win64\chromedriver.exe")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://www.baidu.com')
print(driver.title)
time.sleep(3)
driver.refresh()
driver.quit()

