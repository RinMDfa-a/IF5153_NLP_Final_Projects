import yake

class YAKEExtractor:
    def __init__(self):
        self.extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=3)
    
    def extract(self, text):
        return self.extractor.extract_keywords(text)