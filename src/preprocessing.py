"""
Preprocessing Module
Chunks documentation and adds metadata.
"""

# Standard library imports
import json

# Third-party imports
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports
import config

class DocPreprocessor:

    """

    A class responsible for preprocessing raw documentation files.

    This class loads raw JSON documentation files, splits them into manageable chunks

    using a recursive character text splitter, adds relevant metadata to each chunk,

    and saves the processed chunks to a JSON file for further use in retrieval systems.

    Attributes:

        raw_dir (Path): Path to the directory containing raw data files.

        processed_dir (Path): Path to the directory for processed data.

        text_splitter (RecursiveCharacterTextSplitter): Configured text splitter instance.

    """

    def __init__(self):

        # Set the path to the directory containing raw data files
        self.raw_dir = config.RAW_DATA_DIR

        # Set the path to the directory for processed data and ensure it exists
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the text splitter for chunking documents
        # Uses recursive splitting with specified chunk size, overlap, and separators
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP, length_function=len, separators=["\n\n", "\n", " ", ""])

    def load_raw_docs(self):

        """Loads all raw JSON documentation files from the specified directory.

        Iterates through predefined JSON files, loads their content, and aggregates

        them into a single list of documents. Handles both list and single object JSON structures.

        Returns:

            list: A list of document dictionaries loaded from the JSON files.

        """

        # Initialize an empty list to hold all loaded documents
        docs = []

        # List of JSON files to load
        json_files = ['python_stdlib.json', 'requests_docs.json', 'pandas_docs.json']

        # Iterate over each file
        for filename in json_files:

            # Construct the full path to the file
            file_path = self.raw_dir / filename

            # Check if the file exists; if not, warn and skip
            if not file_path.exists():
                print(f"⚠️  Warning: {filename} not found, skipping...")
                continue

            # Open and load the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)

                # If the JSON is a list, extend the docs list; otherwise, append as single item
                if isinstance(json_data, list):
                    docs.extend(json_data)
                else:
                    docs.append(json_data)
        

        # Print the number of loaded documents
        print(f"✅ Loaded {len(docs)} raw documents")
        return docs
    
    def chunk_documents(self, docs):

        """Chunks the provided documents into smaller pieces and adds metadata to each chunk.

        For each document, splits the text into chunks using the configured text splitter,

        then creates a dictionary for each chunk containing unique IDs, source information,

        and the chunk text.

        Args:

            docs (list): List of document dictionaries, each containing 'text', 'source', 'url', etc.

        Returns:

            list: List of dictionaries, each representing a chunk with metadata.

        """

        # List to hold all chunk data
        all_chunks = []

        # Global counter for unique chunk IDs across all documents
        global_chunk_id = 0

        # Iterate over each document with a progress bar
        for doc in tqdm(docs, desc="chunking documents"):

            # Extract the text content from the document
            text = doc['text']

            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)

            # For each chunk, create metadata dictionary
            for i, chunk_text in enumerate(chunks):

                # Build the chunk data dictionary
                chunk_data = {
                    "chunk_id":global_chunk_id,
                    "local_chunk_id":i,
                    "source":doc['source'],
                    "module":doc.get("module") or doc.get("page"),
                    "url":doc['url'],
                    "text":chunk_text
                }

                # Increment global chunk ID
                global_chunk_id += 1

                # Append the chunk data to the list
                all_chunks.append(chunk_data)
        

        return all_chunks
    
    def save_chunks(self, all_chunks):

        """Saves the list of processed chunks to a JSON file.

        Writes the chunks to 'all_chunks.json' in the processed directory with

        proper indentation and UTF-8 encoding to preserve special characters.

        Args:

            all_chunks (list): List of chunk dictionaries to save.

        """

        # Define the output file path
        output_file = self.processed_dir / "all_chunks.json"

        # Open the file and dump the JSON data
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(all_chunks, file, indent=4, ensure_ascii=False)
        

        # Confirm the save operation
        print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")


    def process_all_docs(self):

        """Executes the complete preprocessing pipeline.

        This method orchestrates the entire preprocessing workflow:

        1. Loads raw documentation files

        2. Chunks the documents and adds metadata

        3. Saves the processed chunks to a file

        4. Displays summary statistics

        """

        # Announce the start of preprocessing
        print("🚀 Starting preprocessing...")

        # Load raw docs
        docs = self.load_raw_docs()

        # Chunk and add metadata
        all_chunks = self.chunk_documents(docs)

        # Save
        self.save_chunks(all_chunks)

        # Display processing statistics
        print(f"\n{'='*60}")
        print(f"✅ Preprocessing complete!")
        print(f"{'='*60}")
        print(f"📊 Statistics:")
        print(f"   Documents processed: {len(docs)}")
        print(f"   Total chunks created: {len(all_chunks)}")
        print(f"   Average chunks per doc: {len(all_chunks) / len(docs):.1f}")
        print(f"   Output file: {self.processed_dir / 'all_chunks.json'}")
        print(f"{'='*60}")

def main():

    """Main entry point for the preprocessing script.

    Instantiates the DocPreprocessor and runs the full processing pipeline.

    """

    # Create an instance of the document preprocessor
    doc_processor = DocPreprocessor()

    # Execute the preprocessing pipeline
    doc_processor.process_all_docs()

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
