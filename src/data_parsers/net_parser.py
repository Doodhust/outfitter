from icrawler.builtin import GoogleImageCrawler

google_Crawler = GoogleImageCrawler(storage={'root_dir': '/home/doodhust/sourse/data'})

# Определяем ключевые слова и источник
keyword = 'dresses'
source = 'fashiers.com'

# Создаем фильтр для поиска изображений с указанным источником
filters = dict(
    source=f'.*{source}.*'
)

# Запускаем загрузку изображений с применением фильтра
google_Crawler.crawl(keyword=keyword, max_num=5, filters=filters)