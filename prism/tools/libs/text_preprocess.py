import warnings
from typing import List, Literal

from prism.tools.tool_registry import register_tool

TEXT_TAGS = ["text processing", "natural language processing"]

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    warnings.warn("NLTK not available. Some text processing tools will not work.")


@register_tool(tags=TEXT_TAGS)
class TextSplitter:
    """
    Split text into smaller chunks based on various strategies.
    """
    
    def split_text(self, text: str, strategy: Literal["sentence", "word", "character", "paragraph"] = "sentence",
                   chunk_size: int = 100, overlap: int = 0) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text (str): Input text
            strategy (str): Splitting strategy - "sentence", "word", "character", "paragraph"
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if strategy in ["sentence", "word"] and not HAS_NLTK:
            warnings.warn("NLTK not available. Some text splitting features may not work.")
            
        if strategy == "sentence":
            if HAS_NLTK:
                sentences = sent_tokenize(text)
                chunks = self._create_chunks(sentences, strategy, chunk_size, overlap)
            else:
                # Fallback to simple sentence splitting
                sentences = text.split('. ')
                chunks = self._create_chunks(sentences, strategy, chunk_size, overlap)
                
        elif strategy == "word":
            if HAS_NLTK:
                words = word_tokenize(text)
            else:
                words = text.split()
            chunks = self._create_word_chunks(words, chunk_size, overlap)
            
        elif strategy == "character":
            chunks = self._create_char_chunks(text, chunk_size, overlap)
            
        elif strategy == "paragraph":
            paragraphs = text.split('\n\n')
            chunks = self._create_chunks(paragraphs, strategy, chunk_size, overlap)
            
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        
        return chunks
    
    def _create_chunks(self, items: List[str], strategy: str, chunk_size: int, overlap: int) -> List[str]:
        """Create chunks from list of items."""
        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(items), step):
            chunk_items = items[i:i + chunk_size]
            if strategy == "sentence":
                chunk = '. '.join(chunk_items)
            elif strategy == "paragraph":
                chunk = '\n\n'.join(chunk_items)
            else:
                chunk = ' '.join(chunk_items)
            chunks.append(chunk)
        
        return chunks
    
    def _create_word_chunks(self, words: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Create chunks from words."""
        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        
        return chunks
    
    def _create_char_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create chunks from characters."""
        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks


@register_tool(tags=TEXT_TAGS)
class TextTokenizer:
    """
    Tokenize text using various strategies.
    """
    
    def tokenize(self, text: str, tokenize_type: Literal["word", "sentence", "subword"] = "word") -> List[str]:
        """
        Tokenize text.
        
        Args:
            text (str): Input text
            tokenize_type (str): Type of tokenization - "word", "sentence", "subword"
            
        Returns:
            List[str]: List of tokens
        """
        if tokenize_type in ["word", "sentence"] and not HAS_NLTK:
            warnings.warn("NLTK not available. Some tokenization features may not work.")
            
        if tokenize_type == "word":
            if HAS_NLTK:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()
                
        elif tokenize_type == "sentence":
            if HAS_NLTK:
                tokens = sent_tokenize(text)
            else:
                tokens = text.split('. ')
                
        elif tokenize_type == "subword":
            # Simple subword tokenization (can be extended with BPE, etc.)
            tokens = []
            for word in text.split():
                if len(word) > 6:
                    # Split long words into subwords
                    for i in range(0, len(word), 4):
                        tokens.append(word[i:i+4])
                else:
                    tokens.append(word)
        else:
            raise ValueError(f"Unknown tokenization type: {tokenize_type}")
        
        return tokens
