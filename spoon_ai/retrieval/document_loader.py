from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    CSVLoader, 
    JSONLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Callable, Type, Union
import os
import logging
import glob as glob_module

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Register supported file types and their corresponding loaders
        self.extension_loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            # JSON requires special handling
            ".json": lambda file_path: JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
        }
    
    def load_directory(self, directory_path: str, glob_pattern: Optional[str] = None) -> List[Document]:
        """Load documents from a directory"""
        # Check if the path is a file instead of a directory
        if os.path.isfile(directory_path):
            return self.load_file(directory_path)
            
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        
        # If a specific glob pattern is provided, use it to load documents
        if glob_pattern:
            try:
                loader = DirectoryLoader(
                    directory_path, 
                    glob=glob_pattern,
                    loader_cls=self._get_loader_for_extension(glob_pattern)
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents matching pattern {glob_pattern} from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading documents with pattern {glob_pattern}: {e}")
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            return split_docs
        
        # Otherwise, automatically detect files in the directory and load them
        try:
            # Get all files in the directory
            all_files = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            # Group files by extension
            files_by_extension = {}
            for file_path in all_files:
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()  # Convert to lowercase to ensure matching
                if ext in self.extension_loaders:
                    if ext not in files_by_extension:
                        files_by_extension[ext] = []
                    files_by_extension[ext].append(file_path)
            
            # Load each type of file
            for ext, files in files_by_extension.items():
                logger.info(f"Found {len(files)} {ext} files")
                loader_cls = self.extension_loaders[ext]
                
                # Use appropriate loader for each file
                for file_path in files:
                    try:
                        if callable(loader_cls) and not isinstance(loader_cls, type):
                            # For factory functions (like JSON loader)
                            loader = loader_cls(file_path)
                        else:
                            # For regular loader classes
                            loader = loader_cls(file_path)
                        
                        file_docs = loader.load()
                        documents.extend(file_docs)
                        logger.info(f"Loaded document: {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
            
            logger.info(f"Loaded {len(documents)} documents in total")
            
        except Exception as e:
            logger.error(f"Error during automatic file detection: {e}")
        
        # If no files were found, try using standard DirectoryLoader methods
        if not documents:
            logger.info("No files detected automatically, falling back to standard loaders")
            
            # Load text files
            try:
                loader = DirectoryLoader(
                    directory_path, 
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                text_docs = loader.load()
                documents.extend(text_docs)
                logger.info(f"Loaded {len(text_docs)} text documents from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading text documents: {e}")
            
            # Load PDF files
            try:
                loader = DirectoryLoader(
                    directory_path, 
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                logger.info(f"Loaded {len(pdf_docs)} PDF documents from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading PDF documents: {e}")
                
           
                
            # Load CSV files
            try:
                loader = DirectoryLoader(
                    directory_path, 
                    glob="**/*.csv",
                    loader_cls=CSVLoader
                )
                csv_docs = loader.load()
                documents.extend(csv_docs)
                logger.info(f"Loaded {len(csv_docs)} CSV documents from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading CSV documents: {e}")
                
            # Load JSON files
            try:
                # For JSON files, we need a custom loader configuration
                def _json_loader_factory(file_path):
                    return JSONLoader(
                        file_path=file_path,
                        jq_schema=".",
                        text_content=False
                    )
                    
                loader = DirectoryLoader(
                    directory_path, 
                    glob="**/*.json",
                    loader_cls=lambda file_path: _json_loader_factory(file_path)
                )
                json_docs = loader.load()
                documents.extend(json_docs)
                logger.info(f"Loaded {len(json_docs)} JSON documents from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading JSON documents: {e}")
                
            # Load HTML files
            try:
                loader = DirectoryLoader(
                    directory_path, 
                    glob="**/*.html",
                    loader_cls=UnstructuredHTMLLoader
                )
                html_docs = loader.load()
                documents.extend(html_docs)
                logger.info(f"Loaded {len(html_docs)} HTML documents from {directory_path}")
            except Exception as e:
                logger.error(f"Error loading HTML documents: {e}")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        return split_docs
    
    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and return the documents"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
            
        documents = []
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()  # Convert to lowercase to ensure matching
        
        if ext not in self.extension_loaders:
            raise ValueError(f"Unsupported file type: {ext}. Supported types are: {', '.join(self.extension_loaders.keys())}")
            
        try:
            loader_cls = self.extension_loaders[ext]
            
            if callable(loader_cls) and not isinstance(loader_cls, type):
                # For factory functions (like JSON loader)
                loader = loader_cls(file_path)
            else:
                # For regular loader classes
                loader = loader_cls(file_path)
                
            file_docs = loader.load()
            documents.extend(file_docs)
            logger.info(f"Loaded document: {file_path}")
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
        
    def _get_loader_for_extension(self, pattern: str):
        """Get the appropriate loader class for a file extension"""
        # Check if the pattern directly matches a known extension
        for ext, loader in self.extension_loaders.items():
            if pattern.endswith(ext):
                return loader
        
        # If it's a complex glob pattern, try to extract the extension
        # For example "**/*.{txt,pdf}" or "**/*.txt"
        if "{" in pattern and "}" in pattern:
            # Handle patterns like {txt,pdf}
            ext_part = pattern.split(".")[-1]
            if "{" in ext_part and "}" in ext_part:
                # Extract the list of extensions
                exts = ext_part.strip("{}").split(",")
                if exts:
                    # Use the first extension
                    first_ext = f".{exts[0]}"
                    if first_ext in self.extension_loaders:
                        return self.extension_loaders[first_ext]
        
        # Default to TextLoader
        logger.warning(f"No specific loader for pattern {pattern}, using TextLoader")
        return TextLoader 