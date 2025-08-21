import argparse
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions  # 改为Chrome选项
from selenium.webdriver.chrome.service import Service as ChromeService  # 改为Chrome服务
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_chrome(download_dir: str, headless: bool = False) -> webdriver.Chrome:  # 改为Chrome
    options = ChromeOptions()  # 改为Chrome选项
    
    # Chrome特有的预设置
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0,  # 允许下载弹出窗口
    }
    options.add_experimental_option("prefs", prefs)
    
    # Chrome特有参数
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")
    
    # 添加Chrome浏览器兼容性选项
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    if headless:
        options.add_argument("--headless=new")
    
    # 修改为Chrome驱动路径 (根据实际安装位置调整)
    # Windows路径示例: r"C:\bin\chromedriver.exe"
    # macOS/Linux路径示例: "/usr/local/bin/chromedriver"
    service = ChromeService(executable_path=r"C:\Users\Administrator\Downloads\chromedriver-win64\chromedriver.exe")  # 改为Chrome服务
    
    driver = webdriver.Chrome(service=service, options=options)  # 改为Chrome驱动
    driver.set_page_load_timeout(40)
    driver.implicitly_wait(2)

    # Chrome使用不同的下载行为设置
    params = {
        "behavior": "allow",
        "downloadPath": download_dir
    }
    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", params)
    except Exception:
        pass
    return driver

def add_custom_cookies(driver):
    """添加自定义Cookie到浏览器"""
    # 替换为你复制的实际Cookie
    cookie_str = "CASTGC=TGT-190672-eOlqPVnwyzkOCuufLhOMeluuzigRLJp2zBmXa3HIFZdxBmBP4C-auth-iploginservice-585987dd68-9f7xw; WFKS.Auth=%7B%22Context%22%3A%7B%22AccountIds%22%3A%5B%22Group.zgdzdx%22%2C%22Shibboleth.zgdzdx%22%2C%22trical.WFMinerBalanceLimitRetail%22%2C%22ForeignLiterature.zgdzdx%22%2C%22WFKSPeriodicalEn.zgdzdx%22%2C%22WFKSThesisEn.zgdzdx%22%2C%22WFKSNstrEn.zgdzdx%22%2C%22GTimeLimit.zgdzdx%22%5D%2C%22Data%22%3A%5B%7B%22Key%22%3A%22Group.zgdzdx.DisplayName%22%2C%22Value%22%3A%22%E4%B8%AD%E5%9B%BD%E5%9C%B0%E8%B4%A8%E5%A4%A7%E5%AD%A6%E6%AD%A6%E6%B1%89%22%7D%5D%2C%22SessionId%22%3A%22dbc317dd-2eb4-4cf2-ba98-8c60648cf1bc%22%2C%22Sign%22%3A%22PzW8C%2BOyRekHP75I5dAsU4SmxIn6pD2VqnTuxWl6Qc5L%2BNAsbFG%5C%2FEt4lhFhbxnl7%22%7D%2C%22LastUpdate%22%3A%222025-08-14T01%3A03%3A49Z%22%2C%22TicketSign%22%3A%22wtItXQ8ArUv2q2kD2b%5C%2F7cA%3D%3D%22%2C%22UserIp%22%3Anull%7D; x-yit-token=fd9fec91091219c37a39188ec988af9665b37298f9a1076ee5744b06875d30e6; x-yit-token=fd9fec91091219c37a39188ec988af9665b37298f9a1076ee5744b06875d30e6; CLICKIT_SESSION=c371a594-3397-4772-8962-6fc9f8b65e19; yit_sw=true"
    
    # 清除现有会话
    driver.delete_all_cookies()
    
    # 解析并添加Cookie
    for cookie in cookie_str.split(';'):
        if '=' in cookie:
            name, value = cookie.strip().split('=', 1)
            driver.add_cookie({
                'name': name,
                'value': value,
                'domain': '.wanfangdata.com.cn',  # 关键点
                'path': '/',
                'secure': False,
                'httpOnly': False
            })
    print("自定义Cookie注入完成")


# 以下函数保持不变...
def wait(driver: webdriver.Chrome, timeout: int = 15):  # 类型改为Chrome
    return WebDriverWait(driver, timeout)


def smart_click(driver: webdriver.Chrome, el):  # 类型改为Chrome
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.15)
    try:
        el.click()
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", el)
        except Exception:
            pass


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[()\（\）\[\]\【\】\《\》\<\>\"\“\”'、，,。.:：;；\-—_]", "", s)
    return s


def field_matches(text: str, query: str) -> bool:
    return normalize_text(query) in normalize_text(text)


def navigate_to_search(driver: webdriver.Chrome, query: str):  # 类型改为Chrome
    encoded = urllib.parse.quote(query)
    driver.get(f"https://s.wanfangdata.com.cn/paper?q={encoded}")
    add_custom_cookies(driver)  # 添加自定义Cookie
    time.sleep(0.5)
    wait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(1.0)
    print(f"当前结果页: {driver.current_url}")
    try:
        for _ in range(2):
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight/2);")
            time.sleep(0.4)
    except Exception:
        pass


def close_possible_popups(driver: webdriver.Chrome):  # 类型改为Chrome
    texts = ["知道了", "同意", "接受", "关闭", "我知道了", "暂不", "×", "X"]
    for t in texts:
        xp = f"//button[contains(normalize-space(.), '{t}')] | //a[contains(normalize-space(.), '{t}')] | //span[contains(normalize-space(.), '{t}')]"
        els = driver.find_elements(By.XPATH, xp)
        for el in els:
            try:
                if el.is_displayed():
                    smart_click(driver, el)
                    time.sleep(0.1)
            except Exception:
                continue


def locate_result_cards(driver: webdriver.Chrome) -> List[Tuple[object, str, str]]:  # 类型改为Chrome
    close_possible_popups(driver)
    time.sleep(0.2)

    title_nodes = driver.find_elements(By.XPATH,
        "//span[contains(@class,'title')] | //a[contains(@class,'title')]")
    cards = []
    seen = set()

    for tn in title_nodes:
        try:
            try:
                card = tn.find_element(By.XPATH, ".//ancestor::*[contains(@class,'result') or contains(@class,'item') or contains(@class,'record')][1]")
            except Exception:
                card = tn
            text = (card.text or tn.text or "").strip()
            href = ""
            try:
                a = card.find_element(By.XPATH, ".//a[starts-with(@href,'https://d.wanfangdata.com.cn/')]")
                href = a.get_attribute("href") or ""
            except Exception:
                pass
            if not href:
                for attr in ["data-url", "data-href", "data-link", "data-target", "data-weburl", "onclick"]:
                    try:
                        v = (card.get_attribute(attr) or "").strip()
                        m = re.search(r"https?://d\.wanfangdata\.com\.cn/[\w/%.+-]+", v)
                        if m:
                            href = m.group(0)
                            break
                    except Exception:
                        continue
            key = (text, href)
            if not text or key in seen:
                continue
            seen.add(key)
            cards.append((card, text, href))
        except Exception:
            continue

    if not cards:
        abs_links = driver.find_elements(By.XPATH, "//a[starts-with(@href,'https://d.wanfangdata.com.cn/')]")
        for a in abs_links:
            href = a.get_attribute("href") or ""
            txt = (a.text or "").strip()
            if href:
                cards.append((a, txt, href))
    return cards


def open_in_new_tab(driver: webdriver.Chrome, url: str):  # 类型改为Chrome
    driver.execute_script("window.open(arguments[0], '_blank');", url)
    driver.switch_to.window(driver.window_handles[-1])


def maybe_wait_for_login(driver: webdriver.Chrome):  # 类型改为Chrome
    return


def _download_clicks(driver: webdriver.Chrome) -> bool:  # 类型改为Chrome
    labels = ["PDF下载", "下载PDF", "PDF", "全文下载", "下载"]
    for lb in labels:
        try:
            btn = driver.find_element(By.XPATH, f'//a[contains(normalize-space(.), "{lb}")] | //button[contains(normalize-space(.), "{lb}")]')
            smart_click(driver, btn)
            time.sleep(1.0)
            return True
        except Exception:
            continue
    try:
        link = driver.find_element(By.XPATH, "//a[contains(translate(@href,'PDF','pdf'), '.pdf')]")
        smart_click(driver, link)
        time.sleep(1.0)
        return True
    except Exception:
        pass
    try:
        caj = driver.find_element(By.XPATH, "//a[contains(., 'CAJ') or contains(., '下载全文') or contains(., '全文浏览')]")
        smart_click(driver, caj)
        time.sleep(1.0)
        return True
    except Exception:
        pass
    return False


def download_pdf_on_detail(driver: webdriver.Chrome, detail_url: str) -> bool:  # 类型改为Chrome
    open_in_new_tab(driver, detail_url)
    try:
        wait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(0.6)
        maybe_wait_for_login(driver)
        return _download_clicks(driver)
    finally:
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])


def download_on_current_detail(driver: webdriver.Chrome) -> bool:  # 类型改为Chrome
    try:
        wait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        pass
    time.sleep(0.4)
    maybe_wait_for_login(driver)
    return _download_clicks(driver)


def click_into_detail(driver: webdriver.Chrome, card, title_fallback: str) -> bool:  # 类型改为Chrome
    before_handles = driver.window_handles[:]
    before_url = driver.current_url
    target = None
    try:
        target = card.find_element(By.XPATH, ".//span[contains(@class,'title')] | .//a[contains(@class,'title')]")
    except Exception:
        target = card
    smart_click(driver, target)
    for _ in range(30):
        time.sleep(0.2)
        after = driver.window_handles[:]
        if len(after) > len(before_handles):
            driver.switch_to.window(after[-1])
            return True
        cur = (driver.current_url or "")
        if "d.wanfangdata.com.cn" in cur and cur != before_url:
            return True
    try:
        anc = target.find_element(By.XPATH, ".//ancestor::*[@onclick or @role='link' or contains(@class,'item') or contains(@class,'record')][1]")
        smart_click(driver, anc)
        for _ in range(20):
            time.sleep(0.2)
            after = driver.window_handles[:]
            if len(after) > len(before_handles):
                driver.switch_to.window(after[-1])
                return True
            cur = (driver.current_url or "")
            if "d.wanfangdata.com.cn" in cur and cur != before_url:
                return True
    except Exception:
        pass
    if title_fallback:
        try:
            xp = ("(//span[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//a[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//div[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//div[contains(normalize-space(.), $t])[1]").replace("$t", f"\"{title_fallback}\"")
            el = driver.find_element(By.XPATH, xp)
            smart_click(driver, el)
            for _ in range(20):
                time.sleep(0.2)
                after = driver.window_handles[:]
                if len(after) > len(before_handles):
                    driver.switch_to.window(after[-1])
                    return True
                cur = (driver.current_url or "")
                if "d.wanfangdata.com.cn" in cur and cur != before_url:
                    return True
        except Exception:
            pass
    return False


def go_next_page(driver: webdriver.Chrome) -> bool:  # 类型改为Chrome
    import re
    try:
        next_arrow = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'bottom-pagination')]//span[contains(@class,'next') and not(contains(@style,'display: none'))]"
        )
        smart_click(driver, next_arrow)
        time.sleep(1.0)
        return True
    except Exception:
        pass

    def to_int(txt: str):
        m = re.search(r"\d+", (txt or "").strip())
        return int(m.group(0)) if m else None

    current_num = None
    try:
        curr_el = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'bottom-pagination')]//span[contains(@class,'pager') and contains(@class,'active')]"
        )
        current_num = to_int(curr_el.text or curr_el.get_attribute("textContent"))
    except Exception:
        current_num = None

    pager_els = driver.find_elements(
        By.XPATH,
        "//div[contains(@class,'bottom-pagination')]//span[contains(@class,'pager')]"
    )
    pages = []
    for el in pager_els:
        n = to_int(el.text or el.get_attribute("textContent"))
        if n is not None:
            pages.append((n, el))
    if not pages:
        return False

    pages.sort(key=lambda x: x[0])
    target = None
    if current_num is not None:
        for n, el in pages:
            if n > current_num:
                target = el
                break
    if target is None and pages:
        target = pages[0][1]
    if not target:
        return False
    try:
        smart_click(driver, target)
        time.sleep(1.0)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="陕西延长石油(集团)有限责任公司")
    parser.add_argument("--out", default=str(Path.cwd() / "wanfang_pdfs"))
    parser.add_argument("--match", choices=["exact", "contains"], default="contains")
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--per-page-limit", type=int, default=50)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 改为使用create_chrome函数
    driver = create_chrome(download_dir=out_dir, headless=args.headless)

    try:
        print(f'运行参数: --query "{args.query}" --match {args.match} --max-pages {args.max_pages} --out "{args.out}" --headless {args.headless}')
        navigate_to_search(driver, args.query)
        current_page = 1

        while True:
            print(f"处理第 {current_page} 页结果 ...")
            time.sleep(0.5)
            cards = locate_result_cards(driver)
            print(f"本页检索到候选 {len(cards)} 条")

            selected = []
            for card, text, href in cards:
                if field_matches(text, args.query):
                    selected.append((card, text, href))
            if not selected:
                selected = cards[: min(args.per_page_limit, len(cards))]

            print(f"拟打开 {len(selected)} 条详情进行下载 ...")

            for card, text, href in selected:
                try:
                    if href and href.startswith("https://d.wanfangdata.com.cn/"):
                        ok = download_pdf_on_detail(driver, href)
                        if ok:
                            time.sleep(0.8)
                    else:
                        if click_into_detail(driver, card, text):
                            try:
                                ok = download_on_current_detail(driver)
                                if ok:
                                    time.sleep(0.8)
                            finally:
                                if len(driver.window_handles) > 1:
                                    driver.close()
                                    driver.switch_to.window(driver.window_handles[0])
                                else:
                                    driver.back()
                                    wait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                                    time.sleep(0.5)
                except Exception as e:
                    print(f"下载失败: {e}")

            if current_page >= args.max_pages:
                break
            if not go_next_page(driver):
                print("没有更多分页")
                break
            current_page += 1

        print(f"完成。PDF 已保存于: {out_dir}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()