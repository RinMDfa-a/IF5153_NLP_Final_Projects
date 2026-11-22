from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class OntologyMapper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.ui_terms = ["interface", "screen", "menu", "ui", "hud", "controls", "button layout", "camera", "keyboard", "controller"]
        self.perf_terms = ["fps", "lag", "ping", "latency", "crash", "bug", "stutter", "server", "optimization", "runs perfect"]
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
        
    def get_sentiment(self, text,aspect=None):
        target_text = text
        
        if aspect:
            sentences = re.split(r'[.!?]+', target_text)
            for sentence in sentences:
                if aspect.lower() in sentence.lower():
                    target_text = sentence
                    break
            
        score = self.analyzer.polarity_scores(target_text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def get_opinion(self, review_text):
        text_lower = str(review_text).lower()
        for word in self.opinion_words:
            if re.search(rf"\b{re.escape(word)}\b", text_lower):
                return word
        return "notable"
    
    