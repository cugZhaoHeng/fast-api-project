import os
import re
import time
import json
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class WanfangDownloader:
    def __init__(self, cookie_str: str, download_dir: str, headless: bool = True):
        self.cookie_str = cookie_str
        self.download_dir = download_dir
        self.headless = headless
        self.driver = self._create_driver()
        self._inject_cookies()
    
    def _create_driver(self) -> webdriver.Chrome:
        """创建并配置Chrome浏览器实例"""
        options = Options()
        
        # 设置下载选项
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0
        }
        options.add_experimental_option("prefs", prefs)
        
        # 添加浏览器参数
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 创建驱动服务 (修改为您的chromedriver路径)
        service = Service(executable_path=r"C:\Users\Administrator\Downloads\chromedriver-win64\chromedriver.exe")
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(40)
        driver.implicitly_wait(2)
        
        # 设置下载行为
        try:
            driver.execute_cdp_cmd("Page.setDownloadBehavior", {
                "behavior": "allow",
                "downloadPath": self.download_dir
            })
        except Exception:
            pass
        
        return driver
    
    def _inject_cookies(self):
        """注入用户提供的Cookie"""
        # 先访问域名以设置Cookie作用域
        self.driver.get("https://s.wanfangdata.com.cn/")
        
        # 清除现有Cookie
        self.driver.delete_all_cookies()
        
        # 解析并添加Cookie
        for cookie in self.cookie_str.split(';'):
            cookie = cookie.strip()
            if not cookie:
                continue
                
            name, value = cookie.split('=', 1)
            self.driver.add_cookie({
                'name': name,
                'value': value,
                'domain': '.wanfangdata.com.cn',  # 关键：使用点开头的顶级域名
                'path': '/',
                'secure': False,
                'httpOnly': False
            })
        
        print("✅ Cookie已成功注入")
        self.driver.refresh()  # 刷新使Cookie生效
        time.sleep(2)
    
    def navigate_to_search(self, query: str):
        """导航到搜索结果页"""
        encoded = urllib.parse.quote(query)
        url = f"https://s.wanfangdata.com.cn/paper?q={encoded}"
        self.driver.get(url)
        
        # 等待页面加载
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(1)
        print(f"🔍 搜索关键词: {query}")
        print(f"🌐 当前结果页: {self.driver.current_url}")
    
    def _close_popups(self):
        """关闭可能的弹窗"""
        texts = ["知道了", "同意", "接受", "关闭", "我知道了", "暂不", "×", "X"]
        for t in texts:
            try:
                xp = f"//button[contains(., '{t}')] | //a[contains(., '{t}')] | //span[contains(., '{t}')]"
                elements = self.driver.find_elements(By.XPATH, xp)
                for el in elements:
                    if el.is_displayed():
                        self._smart_click(el)
                        time.sleep(0.2)
                        print(f"  已关闭弹窗: {t}")
            except Exception:
                continue
    
    def _smart_click(self, element):
        """更可靠的点击方法"""
        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
        time.sleep(0.2)
        try:
            element.click()
        except Exception:
            try:
                self.driver.execute_script("arguments[0].click();", element)
            except Exception:
                pass
    
    def _get_result_cards(self) -> List[Tuple[str, str]]:
        """获取当前页面的结果卡片"""
        self._close_popups()
        time.sleep(0.5)
        
        cards = []
        # 尝试多种定位方式获取结果项
        try:
            # 方式1: 通过卡片容器定位
            items = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "div[class*='result'], div[class*='item'], div[class*='record']"
            )
            
            for item in items:
                try:
                    # 获取标题文本
                    title = item.find_element(
                        By.CSS_SELECTOR, "span[class*='title'], a[class*='title']"
                    ).text.strip()
                    
                    # 获取详情页链接
                    link = ""
                    try:
                        a_tag = item.find_element(
                            By.CSS_SELECTOR, "a[href*='d.wanfangdata.com.cn']"
                        )
                        link = a_tag.get_attribute("href")
                    except Exception:
                        pass
                    
                    if title and link:
                        cards.append((title, link))
                except Exception:
                    continue
        except Exception:
            pass
        
        # 方式2: 直接通过链接定位（备用方法）
        if not cards:
            try:
                links = self.driver.find_elements(
                    By.CSS_SELECTOR, "a[href*='d.wanfangdata.com.cn']"
                )
                for link in links:
                    try:
                        title = link.text.strip()
                        href = link.get_attribute("href")
                        if title and href:
                            cards.append((title, href))
                    except Exception:
                        continue
            except Exception:
                pass
        
        print(f"📄 本页找到 {len(cards)} 篇文章")
        return cards
    
    def _download_pdf(self, detail_url: str) -> bool:
        """在新标签页中下载PDF"""
        # 保存当前窗口句柄
        main_window = self.driver.current_window_handle
        
        # 打开新标签页
        self.driver.execute_script(f"window.open('{detail_url}');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        
        try:
            # 等待页面加载
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)
            
            # 尝试查找并点击下载按钮
            download_attempted = False
            download_selectors = [
                "//a[contains(., 'PDF下载')]",
                "//a[contains(., '下载PDF')]",
                "//a[contains(., 'PDF')]",
                "//a[contains(., '全文下载')]",
                "//a[contains(., '下载')]",
                "//a[contains(@href, '.pdf')]"
            ]
            
            for selector in download_selectors:
                try:
                    download_btn = self.driver.find_element(By.XPATH, selector)
                    if download_btn.is_displayed():
                        self._smart_click(download_btn)
                        print("  正在下载PDF...")
                        download_attempted = True
                        time.sleep(2)
                        break
                except Exception:
                    continue
            
            return download_attempted
        finally:
            # 关闭当前标签页并返回主窗口
            self.driver.close()
            self.driver.switch_to.window(main_window)
            time.sleep(0.5)
    
    def _go_next_page(self) -> bool:
        """尝试翻到下一页"""
        try:
            # 尝试查找下一页按钮
            next_buttons = [
                "//button[contains(., '下一页')]",
                "//a[contains(., '下一页')]",
                "//span[contains(@class, 'next')]",
                "//a[contains(@class, 'next')]"
            ]
            
            for selector in next_buttons:
                try:
                    next_btn = self.driver.find_element(By.XPATH, selector)
                    if next_btn.is_displayed():
                        self._smart_click(next_btn)
                        time.sleep(2)
                        print("➡️ 已翻到下一页")
                        return True
                except Exception:
                    continue
            
            print("⏹️ 没有找到下一页按钮")
            return False
        except Exception:
            return False
    
    def download_papers(self, query: str, max_pages: int = 5):
        """主下载函数"""
        # 导航到搜索结果页
        self.navigate_to_search(query)
        
        downloaded_count = 0
        current_page = 1
        
        while current_page <= max_pages:
            print(f"\n📖 处理第 {current_page} 页...")
            
            # 获取本页所有结果
            cards = self._get_result_cards()
            
            # 下载每篇文章
            for title, link in cards:
                try:
                    print(f"  文章: {title[:40]}...")
                    print(f"  链接: {link}")
                    
                    if self._download_pdf(link):
                        downloaded_count += 1
                        print(f"  下载成功! (总计: {downloaded_count})")
                    
                    # 避免请求过快
                    time.sleep(1.5)
                except Exception as e:
                    print(f"  下载失败: {str(e)}")
            
            # 尝试翻页
            if not self._go_next_page():
                break
                
            current_page += 1
        
        print(f"\n✅ 下载完成! 共下载 {downloaded_count} 篇论文")
    
    def close(self):
        """关闭浏览器"""
        self.driver.quit()
        print("🛑 浏览器已关闭")


def main():
    parser = argparse.ArgumentParser(description="万方数据论文批量下载工具")
    parser.add_argument("--cookie", required=True, help="x-yit-token=fd9fec91091219c37a39188ec988af9665b37298f9a1076ee5744b06875d30e6; CLICKIT_SESSION=c371a594-3397-4772-8962-6fc9f8b65e19; yit_sw=true")
    parser.add_argument("--query", required=True, help="延长油田")
    parser.add_argument("--out", default="wanfang_pdfs", help="下载目录 (默认: wanfang_pdfs)")
    parser.add_argument("--pages", type=int, default=5, help="最大翻页数 (默认: 5)")
    parser.add_argument("--visible", action="store_true", help="显示浏览器界面 (默认隐藏)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    download_dir = os.path.abspath(args.out)
    os.makedirs(download_dir, exist_ok=True)
    print(f"📂 下载目录: {download_dir}")
    
    # 创建下载器实例
    downloader = WanfangDownloader(
        cookie_str=args.cookie,
        download_dir=download_dir,
        headless=not args.visible
    )
    
    try:
        # 执行下载
        downloader.download_papers(
            query=args.query,
            max_pages=args.pages
        )
    finally:
        # 确保浏览器关闭
        downloader.close()


if __name__ == "__main__":
    main()