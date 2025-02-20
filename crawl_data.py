import asyncio
from crawl4ai import *

async def process_result(result):
    with open('./rag_dataset/viettel_dataset.txt', 'a', encoding='utf-8') as f:
        f.write(result.markdown)

base_url = 'https://dangkyviettel.com.vn/'
async def get_urls():
    ans = []
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url= base_url,
        )
        for d in result.links['internal']:
            ans.append(d['href'])
    return ans

async def crawl_single(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        if result.success:
            await process_result(result)
        else:
            print(f"Failed to crawl {result.url}: {result.error_message}")

async def crawl_batch(urls):
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False  
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10,
        monitor=CrawlerMonitor(
            display_mode=DisplayMode.DETAILED
        )
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher
        )

        for result in results:
            if result.success:
                await process_result(result)
            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")

def main():
    urls = asyncio.run(get_urls())
    asyncio.run(crawl_batch(urls))

urls = ['https://4g5gviettel.net/',
        'https://www.thegioididong.com/game-app/tong-hop-cac-goi-3g-4g-viettel-ngay-dung-tien-nhanh-lai-gia-1288179',
        'https://www.dienmayxanh.com/kinh-nghiem-hay/tong-hop-cac-goi-cuoc-hot-3g-4g-viettel-goi-sms-go-1232242',
        'https://viettelmoney.vn/goi-noi-mang-viettel/']
if __name__ == '__main__':
    for url in urls:
        asyncio.run(crawl_single(url))
    main()