import scrapy


'''
With the help of Scrapy one can :

1. Fetch millions of data efficiently
2. Run it on server
3. Fetching data
4. Run spider in multiple processes
'''
# Create the spider class

class ExtractUrls(scrapy.Spider):
    # this name must be unique always
    name = 'extract'

    # function to be invoked
    def start_requests(self):
        # enter urls here
        urls = ['https://www.geeksforgeeks.org/',]

        for url in urls:
            yield scrapy.Request(url, callback=self.parse)

    # parse function
    def parse(self,response):
        # extract features to get title
        titles = response.css('tittle::text').extract()

        # get anchor tags
        links = response.css('a::attr(href)').extract()

        for link in links:
            for title in titles:
                yield {
                    'title': title,
                    'links': link,
                }
                # this becomes recursive.
                if 'geekforgeeks' in link:
                    yield scrapy.Request(url=link, callback=self.parse)


# to run this
# scrapy crawl [name_of_spider from class] -o links.json
# -o means output to the file links.json
# 
