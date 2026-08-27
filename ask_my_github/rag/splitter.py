"""Code-aware document splitting."""

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ask_my_github.config import get_settings
from ask_my_github.logging_config import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using language-aware boundaries."""
    settings = get_settings()
    chunks: list[Document] = []
    for document in documents:
        splitter = _splitter_for(
            document.metadata.get("language"),
            settings.chunk_size,
            settings.chunk_overlap,
        )
        chunks.extend(splitter.split_documents([document]))
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def _splitter_for(
    language_key: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    if language_key is None:
        return _plain_splitter(chunk_size, chunk_overlap)
    try:
        language = Language(language_key)
    except ValueError:
        return _plain_splitter(chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _plain_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
