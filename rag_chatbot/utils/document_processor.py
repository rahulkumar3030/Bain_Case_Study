"""
Document Processor Module

Processes documents from supporting_docs/ folder:
1. Loads and chunks documents using hybrid semantic strategy
2. Generates embeddings via Azure OpenAI
3. Stores chunks in ChromaDB vector database
4. Moves processed files to processed_docs/ folder

Usage:
    python -m chatbot.services.document_processor
"""

import os
import asyncio
import logging
import hashlib
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

from azure_client import AzureOpenAIEmbeddingClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_file = os.getenv("CONFIG_YAML_PATH", "config.yaml")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


class DocumentChunker:
    """Handles intelligent document chunking with section detection."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
    
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        This is a custom layout analysis to detect sections in HR documents.
        
        Args:
            text: Document text
            
        Returns:
            List of sections with title, number, and content
        """
        import re
        
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        logger.debug("number of lines:", len(lines))
        for line in lines:
            match = re.match(r'^(\d+)\.\s+([A-Z\s&]+):', line.strip())
            
            if match:
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                section_num = match.group(1)
                section_title = match.group(2).strip()
                logger.debug("section detected:", section_num, section_title)

                current_section = {
                    'section_number': int(section_num),
                    'section_title': section_title,
                    'content': ''
                }
                current_content = [line]
            else:
                if current_section:
                    current_content.append(line)
        
        
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        if not sections:
            sections = [{
                'section_number': 1,
                'section_title': 'Document',
                'content': text
            }]
        
        logger.info(f"Detected {len(sections)} sections")
        return sections
    
    def chunk_section(
        self, 
        section: Dict[str, Any], 
        file_name: str,
        file_path: str
    ) -> List[Document]:
        """
        Chunk a single section into smaller pieces if needed.
        
        Args:
            section: Section dictionary with content
            file_name: Source filename
            file_path: Source file path
            
        Returns:
            List of LangChain Document objects
        """
        content = section['content']
        
        if len(content) <= self.chunk_size:
            chunks = [content]
        else:
            chunks = self.text_splitter.split_text(content)
        
        documents = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{Path(file_name).stem}_s{section['section_number']}_c{idx}"
            
            metadata = {
                'file_name': file_name,
                'file_path': file_path,
                'section_number': section['section_number'],
                'section_title': section['section_title'],
                'chunk_id': chunk_id,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'created_at': datetime.now().isoformat()
            }
            
            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def chunk_document(
        self, 
        text: str, 
        file_name: str,
        file_path: str
    ) -> List[Document]:
        """
        Chunk entire document using section detection.
        
        Args:
            text: Document content
            file_name: Source filename
            file_path: Source file path
            
        Returns:
            List of all document chunks
        """
        sections = self.detect_sections(text)
        all_chunks = []
        
        for section in sections:
            chunks = self.chunk_section(section, file_name, file_path)
            all_chunks.extend(chunks)
        
        logger.debug("chunks are", all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks from {file_name}")
        return all_chunks


class DocumentProcessor:
    """Main processor for document ingestion pipeline."""
    
    def __init__(self, collection_name: Optional[str]):
        """
        Initialize document processor.
        
        Args:
            collection_name: ChromaDB collection name
        """
        config = load_config()
        
        self.base_dir = Path(os.getcwd())
        self.supporting_docs_dir = self.base_dir / config['paths']['supporting_docs_folder']
        self.processed_docs_dir = self.base_dir / config['paths']['processed_docs_folder']
        self.chroma_db_dir = self.base_dir / config['paths']['data_folder'] / config['paths']['chroma_db_folder']
        self.collection_name = collection_name
        
        # Get chunking config
        self.chunk_size = config['rag']['chunk_size']
        self.chunk_overlap = config['rag']['chunk_overlap']
        
        # Create directories if not present
        self.supporting_docs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
        self.embedding_client = AzureOpenAIEmbeddingClient()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Acme HR documents and policies",
                      "index": "hnsw+bm25"}
        )
        
        logger.info(f"Initialized DocumentProcessor (collection: {self.collection_name})")
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    async def embed_and_store_chunk(self, chunk: Document) -> None:
        """
        Generate embedding and store chunk in ChromaDB.
        
        Args:
            chunk: Document chunk with metadata
        """
        try:
            embeddings = await self.embedding_client.create_embeddings_async([chunk.page_content])
            embedding = embeddings[0]
            #logger.debug("embedding is", embedding)
            self.collection.upsert(
                ids=[chunk.metadata['chunk_id']],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata]
            )
            
        except Exception as e:
            logger.error(f"Failed to embed/store chunk {chunk.metadata['chunk_id']}: {e}")
            raise
    
    async def process_single_document(self, file_path: Path) -> int:
        """
        Process a single document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Number of chunks created
        """
        try:
            logger.info(f"Processing file : {file_path.name}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"{file_path.name} is not a UTF-8 text file. Skipping.")
                content = ""
            if content.strip() != "":
                file_hash = self.compute_file_hash(file_path) # checks if there is a chunk content change in file or not. Can be used in future to update specific sections of documents
            
            chunks = self.chunker.chunk_document(content, file_path.name, str(file_path))
            
            for chunk in chunks:
                chunk.metadata['file_hash'] = file_hash
            
            tasks = [self.embed_and_store_chunk(chunk) for chunk in chunks]
            await asyncio.gather(*tasks)
            
            dest_path = self.processed_docs_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            
            logger.info(f"Completed {file_path.name}: {len(chunks)} chunks stored, moved to processed_docs/")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            raise
    
    async def process_all_documents(self) -> Dict[str, int]:
        """
        Process all documents in supporting_docs folder.
        
        Returns:
            Dictionary with processing statistics
        """
        allowed_extensions = ['*.txt'] # will only take txt files for now can use more extensions once the parsers are added
        files = []
        
        for pattern in allowed_extensions:
            files.extend(self.supporting_docs_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} documents to process")
        
        if not files or len(files) == 0:
            logger.info("No documents to process")
            return {'files_processed': 0, 'total_chunks_created': 0}
        
        logger.info(f"Starting processing of {len(files)} documents...")
        
        tasks = [self.process_single_document(file_path) for file_path in files]
        chunk_counts = await asyncio.gather(*tasks)
        
        total_chunks = sum(chunk_counts)
        
        stats = {
            'files_processed': len(files),
            'total_chunks_created': total_chunks
        }
        
        logger.info(f"Processing complete: {stats['files_processed']} files, {stats['total_chunks_created']} chunks")
        return stats


async def main():
    """Main entry point."""
    logger.info("DOCUMENT EMBEDDING PROCESSOR")
    
    processor = DocumentProcessor(collection_name="acme_hr_docs")
    
    logger.debug ("chroma DB local storage location is", processor.chroma_db_dir)
    logger.debug ("chroma DB collection name is", processor.collection_name)
    
    logger.debug("Processing documents")
    results = await processor.process_all_documents()
    
    logger.debug("\nProcessing Results:")
    for key, value in results.items():
        logger.debug(f"  {key}: {value}")
    
    logger.info("chunks have been added to chromaDB collection successfully.")


if __name__ == "__main__":
    asyncio.run(main())