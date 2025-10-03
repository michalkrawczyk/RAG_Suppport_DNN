def is_empty_text(text: str) -> bool:
    """Check if the text is empty or only whitespace"""
    if not text or text.strip() == "":
        return True
    if text.lower() == "nan":
        return True
    return False