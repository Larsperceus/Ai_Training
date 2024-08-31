import os
import docx
import re
import json
import nltk

# Download required NLTK resources
nltk.download('punkt')


def extract_text_from_docx(docx_path):
    """Extracts and returns text from a .docx file."""
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def clean_text(text):
    """Cleans and preprocesses the extracted text."""
    # Remove multiple spaces and newline characters
    text = re.sub(r'\s+', ' ', text)

    # Remove any non-printable characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Optionally convert text to lowercase
    text = text.lower()

    # Remove unwanted punctuation or characters
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)

    return text


def remove_references(text):
    """Removes reference sections from the text."""
    # Regular expression pattern to identify common reference sections
    # Adjust this pattern based on how references are formatted in your documents
    reference_patterns = [
        r'referentie[^\n]*',  # Dutch for 'reference'
        r'\bref(?:erences?)?:?\s*',
        r'bibliography[:\s]*',
        r'\[\d+\]',  # Matches numbered references like [1], [2], etc.
        r'(?:doi|http|https):\S+'  # Matches DOI or URLs
    ]

    for pattern in reference_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text


def split_into_sentences(text):
    """Splits text into sentences."""
    return nltk.sent_tokenize(text)


def split_into_chunks(text, chunk_size=80):
    """Splits text into smaller chunks of a specified size."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in split_into_sentences(text):
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_docx_files(directory, chunk_size):
    """Processes all .docx files in the specified directory."""
    processed_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.docx'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            raw_text = extract_text_from_docx(file_path)
            cleaned_text = clean_text(raw_text)
            text_without_refs = remove_references(cleaned_text)
            chunks = split_into_chunks(text_without_refs, chunk_size)

            # Create a dictionary for each document
            document_data = {
                "filename": filename,
                "chunks": chunks
            }

            processed_data.append(document_data)

    return processed_data


def save_to_json(data, output_file):
    """Saves the processed data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Define the directory containing your .docx files and chunk size
docx_directory = r'C:\Users\larsk\Downloads\AI-Software\All'
output_json_file = 'processed_documents.json'
chunk_size = 80  # Define the maximum number of characters per chunk

# Process the files and save to JSON
processed_documents = process_docx_files(docx_directory, chunk_size)
save_to_json(processed_documents, output_json_file)

print(f"Processed text saved to {output_json_file}")
