def base_text_splitter(text: str, max_words: int = 200):
    """
    Split text into chunks of max_words words
    """
    words = text.replace("\n", "").split(" ")
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))

    return chunks
