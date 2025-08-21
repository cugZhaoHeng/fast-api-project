import argparse
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_edge(download_dir: str, headless: bool = False) -> webdriver.Edge:
    options = EdgeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "safebrowsing.enabled": True
    })
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")

    service = EdgeService(executable_path=r"C:\bin\msedgedriver.exe")
    driver = webdriver.Edge(service=service, options=options)
    driver.set_page_load_timeout(40)
    driver.implicitly_wait(2)

    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": download_dir
        })
    except Exception:
        pass
    return driver


def wait(driver: webdriver.Edge, timeout: int = 15):
    return WebDriverWait(driver, timeout)


def smart_click(driver: webdriver.Edge, el):
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
    # 列表页不只看“标题”，而是看整条卡片文本是否包含关键词
    return normalize_text(query) in normalize_text(text)


def navigate_to_search(driver: webdriver.Edge, query: str):
    # 直接进检索页，避免被 UI 改写为作者/机构
    encoded = urllib.parse.quote(query)
    driver.get(f"https://s.wanfangdata.com.cn/paper?q={encoded}")
    wait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(1.0)
    print(f"当前结果页: {driver.current_url}")
    try:
        for _ in range(2):
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight/2);")
            time.sleep(0.4)
    except Exception:
        pass


def close_possible_popups(driver: webdriver.Edge):
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


def locate_result_cards(driver: webdriver.Edge) -> List[Tuple[object, str, str]]:
    """
    返回 [(卡片元素或标题元素, 整条卡片文本, 可能的详情href)]。
    href 可能为空，后续用文本点击进入详情。
    """
    close_possible_popups(driver)
    time.sleep(0.2)

    # 1) 先找标题节点（span.title / a.title），再向上找卡片容器
    title_nodes = driver.find_elements(By.XPATH,
        "//span[contains(@class,'title')] | //a[contains(@class,'title')]")
    cards = []
    seen = set()

    for tn in title_nodes:
        try:
            # 找到就近的卡片容器（类名包含 result/item/record 的 div），否则用标题自身
            try:
                card = tn.find_element(By.XPATH, ".//ancestor::*[contains(@class,'result') or contains(@class,'item') or contains(@class,'record')][1]")
            except Exception:
                card = tn
            text = (card.text or tn.text or "").strip()
            # 尝试从卡片内提取详情 href（绝对 d.wanfang 链接或 data- 属性/onclick）
            href = ""
            try:
                a = card.find_element(By.XPATH, ".//a[starts-with(@href,'https://d.wanfangdata.com.cn/')]")
                href = a.get_attribute("href") or ""
            except Exception:
                pass
            if not href:
                # data- 属性或 onclick 里藏的绝对链接
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

    # 2) 若还是空，兜底：全页查找 d.wanfang 绝对链接
    if not cards:
        abs_links = driver.find_elements(By.XPATH, "//a[starts-with(@href,'https://d.wanfangdata.com.cn/')]")
        for a in abs_links:
            href = a.get_attribute("href") or ""
            txt = (a.text or "").strip()
            if href:
                cards.append((a, txt, href))
    return cards


def open_in_new_tab(driver: webdriver.Edge, url: str):
    driver.execute_script("window.open(arguments[0], '_blank');", url)
    driver.switch_to.window(driver.window_handles[-1])


def maybe_wait_for_login(driver: webdriver.Edge):
    # 免登录场景：不拦截
    return


def _download_clicks(driver: webdriver.Edge) -> bool:
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


def download_pdf_on_detail(driver: webdriver.Edge, detail_url: str) -> bool:
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


def download_on_current_detail(driver: webdriver.Edge) -> bool:
    try:
        wait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        pass
    time.sleep(0.4)
    maybe_wait_for_login(driver)
    return _download_clicks(driver)


def click_into_detail(driver: webdriver.Edge, card, title_fallback: str) -> bool:
    """没有 href 时，点击标题/容器进入详情；成功则位于详情页（当前或新标签）。"""
    before_handles = driver.window_handles[:]
    before_url = driver.current_url
    # 优先点击标题
    target = None
    try:
        target = card.find_element(By.XPATH, ".//span[contains(@class,'title')] | .//a[contains(@class,'title')]")
    except Exception:
        target = card
    smart_click(driver, target)
    # 等新标签或 URL 切换
    for _ in range(30):
        time.sleep(0.2)
        after = driver.window_handles[:]
        if len(after) > len(before_handles):
            driver.switch_to.window(after[-1])
            return True
        cur = (driver.current_url or "")
        if "d.wanfangdata.com.cn" in cur and cur != before_url:
            return True
    # 失败时尝试点祖先容器
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
    # 再失败，用文本搜索一次
    if title_fallback:
        try:
            xp = ("(//span[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//a[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//div[contains(@class,'title')][contains(normalize-space(.), $t)] | "
                  "//div[contains(normalize-space(.), $t)])[1]").replace("$t", f"\"{title_fallback}\"")
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


def go_next_page(driver: webdriver.Edge) -> bool:
    import re
    # 先点右箭头：span.next 未隐藏
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

    # 数字分页
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

    driver = create_edge(download_dir=out_dir, headless=args.headless)
    try:
        print(f'运行参数: --query "{args.query}" --match {args.match} --max-pages {args.max_pages} --out "{args.out}" --headless {args.headless}')
        navigate_to_search(driver, args.query)
        current_page = 1

        while True:
            print(f"处理第 {current_page} 页结果 ...")
            time.sleep(0.5)
            cards = locate_result_cards(driver)
            print(f"本页检索到候选 {len(cards)} 条")

            # 放宽匹配：用整卡片文本匹配关键词；没有匹配也尝试回退打开若干条
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
                        # 无直链：点击进入详情（新标签或当前页）
                        if click_into_detail(driver, card, text):
                            try:
                                ok = download_on_current_detail(driver)
                                if ok:
                                    time.sleep(0.8)
                            finally:
                                # 关闭详情标签回到列表
                                if len(driver.window_handles) > 1:
                                    driver.close()
                                    driver.switch_to.window(driver.window_handles[0])
                                else:
                                    # 若在当前标签，回退到列表
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