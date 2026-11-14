class OntologyMapper:
    def __init__(self):
        self.ui_terms = ["interface", "screen", "menu", "ui"]
        
        self.positive_words = ["good", "great", "fluid", "responsive", "breathtaking", "immersive", "clean", "intuitive", "smooth", "rewarding"]
        
        self.negative_words = ["bad", "awful", "clunky", "laggy", "tedious", "unresponsive", "predatory", "repetitive", "frustrating", "punishing", "slow", "stuttering"]
        
    def get_category(self, word):
        if any(term in word.lower() for term in self.ui_terms):
            return "UI/UX"
        else:
            return "Gameplay"
        
    def get_sentiment(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in self.positive_words):
            return "Positive"
        elif any(word in text_lower for word in self.negative_words):
            return "Negative"
        else:
            return "Neutral"
    
    def get_opinion(self, review_text):
        for word in self.positive_words + self.negative_words:
            if word in review_text.lower():
                return word
        return "notable"
    
    