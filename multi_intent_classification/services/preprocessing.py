from underthesea import text_normalize, word_tokenize


def word_normalize(sentence: str) -> str:
    normalized_text = text_normalize(sentence)
    return normalized_text

def word_segment(sentence: str) -> str:
    context = word_tokenize(sentence, format="text")
    return context
