from icrawler.builtin import GoogleImageCrawler

google_Crawler = GoogleImageCrawler(storage={'root_dir': '/home/doodhust/sourse/new_summer_project_DNS/data/test'})

# Определяем ключевые слова и источник
keyword = 'clothes without women'


google_Crawler.crawl(keyword=keyword, max_num=10)