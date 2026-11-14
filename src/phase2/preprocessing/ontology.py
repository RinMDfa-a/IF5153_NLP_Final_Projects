from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class OntologyMapper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.ui_terms = ["interface", "screen", "menu", "ui"]
        self.perf_terms = ["fps", "lag", "ping", "latency", "crash", "bug", "stutter", "server", "optimization"]
        self.audio_terms = ["sound", "audio", "music", "voice", "effects", "soundtrack"]
        
        self.opinion_words = [
            "good", "great", "fluid", "responsive", "breathtaking", "immersive", 
            "clean", "intuitive", "rewarding", "world-class", "expressive",
            "bad", "awful", "clunky", "tedious", "frustrating", "predatory", 
            "punishing", "slow", "stuttering", "repetitive", "floaty", "opaque"
        ]
        
    def get_category(self, word):
        if any(term in word.lower() for term in self.ui_terms):
            return "UI/UX"
        elif any(term in word.lower() for term in self.perf_terms):
            return "Performance"
        elif any(term in word.lower() for term in self.audio_terms):
            return "Audio"
        else:
            return "Gameplay"
        
    def get_sentiment(self, text):
        text_lower = text.lower()
        score = self.analyzer.polarity_scores(text_lower)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def get_opinion(self, review_text):
        text_lower = str(review_text).lower()
        for word in self.opinion_words:
            if word in text_lower:
                return word
        return "notable"
    
    