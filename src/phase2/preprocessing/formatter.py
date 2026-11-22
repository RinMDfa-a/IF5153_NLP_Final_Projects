class PromptFormatter:
    @staticmethod
    def format_merge_game(app_name, reviews_list, aspects_list, seq2seq_labels, review_pairs=None):
        return {
            "app_name": app_name,
            "reviews": reviews_list,
            "aspects": aspects_list,
            "target_label": f"Aspects: [{', '.join(seq2seq_labels)}]",
            "review_pairs": review_pairs or []
        }