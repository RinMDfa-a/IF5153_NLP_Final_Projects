import yake
import re
class YAKEExtractor:
    def __init__(self):
        self.extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5)
    
        self.domain_aspects = [
            "combo system", "button layout", "voice acting", "runs perfect",
            "level design", "sound effects", "single player", "open world",
            "gameplay", "graphics", "controls", "music", "soundtrack", "mechanics",
            "co-op", "campaign", "characters", "multiplayer", "interface", "menu",
            "ui", "fps", "lag", "ping", "bugs", "story", "combat", "balance", "price",
            "difficulty", "weapons", "matchmaking", "dlc", "microtransactions",
            "replayability", "atmosphere", "server", "servers", "quests", "pvp"
        ]
        
    def _clean_text(self, text):
        text = re.sub(r"http\S+|www\.\S+", "", str(text))
        text = re.sub(r"♥+\w*|[-_=]{3,}", "", text)
        return text.strip()
    
    def extract(self, text):
        clean_text = self._clean_text(text)
        
        yake_candidates = self.extractor.extract_keywords(clean_text)
        found_texts = []
        seen = set()
        
        for key_word, score in yake_candidates:
            key_word = key_word.lower()
            for aspect in self.domain_aspects:
                if re.search(rf"\b{re.escape(aspect)}\b", key_word) and aspect not in seen:
                    found_texts.append((aspect, score))
                    seen.add(aspect)
                    break
            if len(found_texts) > 2:
                return found_texts
        
        if not found_texts:
            text_lower = clean_text.lower()
            for aspect in self.domain_aspects:
                if re.search(rf"\b{re.escape(aspect)}\b", text_lower):
                    found_texts.append((aspect, 0.0))
                    if len(found_texts) >= 2:
                        break
                
        return found_texts