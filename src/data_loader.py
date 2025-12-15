from pathlib import Path
from typing import List, Any
from langchain_community.document_loader import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loader import Docx2txtLoader
from langchain_community.document_loader.excel import UnstructuredExcelLoader
from langchain_community.document_loader import JOSNLoader


def load_all_documents(data_dir: str) -> List[Any] :
    
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path : {data_path}")
    document = []
    
    pdf_files = list(data_path.glob('**.*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files : { [str(f) for f in pdf_files] }")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_files}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file} : {e}")
        
            
    txt_files = list(data_path.glob('**.*.txt'))
    print(f'[DEBUG] Founded {len(txt_files)} TXT files: {[str(f) for f in txt_files]} ')
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT : {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loaded.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            document.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load TXT {txt_file} : {e}")
    
    
    
def process_all_pdf(pdf_directory):
    
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    pdf_files = list(pdf_dir.glob('**/*.pdf'))
    
    print(f"found {len(pdf_files)} PDF files to Process")
    
    for pdf_files in pdf_files:
        print(f"\npreprocessing : {pdf_files.name}")
        try:
            loader = PyPDFLoader(str(pdf_files))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source_file'] = pdf_files.name
                doc.metadata['file_type'] = 'pdf'
                
            all_documents.extend(documents)
            print(f" Loaded {len(documents)} pages")
        except Exception as e:
            print(f" Error : {e}")
            
    print(f"\n Total documents loaded : {len(all_documents)}")
    return all_documents    

all_pdf_files = process_all_pdf('./data')