from playwright.sync_api import Browser, Page, sync_playwright  # 修正1: インポートの変更


def download_single_html(page: Page, url: str):
    """指定されたURLからHTMLコンテンツをダウンロードする。"""
    try:
        # wait_until="domcontentloaded" はデフォルト値なので省略可能ですが、残しておきます
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        html_content = page.content()
        return html_content
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None  # エラー発生時はNoneを返す


def download(url: str):
    """Playwrightを初期化し、ブラウザを起動してダウンロード処理を行う。"""
    html = None
    # 修正2: sync_playwright() を使用してPlaywrightを初期化（p の役割を果たす）
    with sync_playwright() as p:
        # ブラウザの起動 (p.firefox の部分)
        # withブロックで囲むことで、例外発生時も確実にブラウザを閉じます
        browser: Browser = p.firefox.launch(headless=True)
        try:
            page = browser.new_page()
            html = download_single_html(page, url)
        finally:
            # 修正3: browser.close() を明示的に呼ぶ（withブロックにより不要な場合もあるが、ここでは明確化のため）
            # 上記の with sync_playwright() as p: と with browser: のようにすることで、
            # .close() の呼び出しを省略できますが、ここでは分かりやすさのため try...finally を使用
            # browser.close() は with browser: を使わない場合に必要
            browser.close()  # withブロックの外に出すとエラーになる可能性があるため、try...finallyブロック内に配置するのが安全

    return html


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # コマンドライン引数からURLを取得
        url = sys.argv[1]
        print(f"Downloading HTML from: {url}")
        html_content = download(url)
        if html_content:
            print("\n--- HTML Content Snippet ---")
            print(html_content[:500])  # 最初の500文字だけ表示
            print("...")
        else:
            print(f"Failed to download HTML from: {url}")
    else:
        print("Usage: python your_script_name.py <url>")
