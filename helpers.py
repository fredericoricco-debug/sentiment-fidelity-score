def clean_text(text):
    # Clean the text in various ways
    text = text.strip() # Remove leading/trailing whitespace
    text = text.replace("\n", " ") # Replace newlines with spaces
    text = text.replace("\"", "") # Remove double quotes
    text = text.replace("  ", " ") # Replace double spaces with single spaces
    text = text.lower() # Convert to lowercase
    # Remove non-english characters
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text