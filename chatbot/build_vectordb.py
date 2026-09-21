"""Rebuild the Probahini vector store from the source PDF, with chunking.

Run during the Docker build, so the store shipped in the image is always
generated from mergedd_bn_en.pdf rather than committed as binary files.

WHY THIS EXISTS
---------------
Probahini.ipynb, which built the committed store, imports
RecursiveCharacterTextSplitter on its second line and then never calls it:

    loader = PyPDFLoader('mergedd_bn_en.pdf')
    doc = loader.load()                      # one Document per PDF PAGE
    ...
    for i, text in enumerate(doc):
        collection.add(documents=[text.page_content], ...)   # a whole page

So every entry in the store is an entire page of a Q&A table — dozens of
unrelated question-and-answer rows in one lump. chain.py then asks for the
three closest entries and pastes them into the prompt, which is why a single
question ships ~18,000 tokens to the model: three whole pages, of which
perhaps two rows are relevant.

Measured on one Bangla question: 17,978 prompt tokens, 29,598 characters
retrieved.

Chunking fixes that at the source. Each entry becomes a row or two, so the
same three-to-six results are all on topic and a fraction of the size.

EMBEDDINGS
----------
Documents are added without precomputed embeddings, so Chroma embeds them with
its own default model — the same all-MiniLM-L6-v2 that chain.py's
`query_texts=` uses at query time. The notebook computed embeddings separately
through sentence-transformers, which happened to match; letting one side do
both removes the chance of them drifting apart.

It also means the 79MB ONNX model is downloaded here, during the build, and
baked into the image. At runtime the free instance is destroyed whenever it
sleeps, so without this it re-downloads that model on the first question after
every idle period.
"""

import shutil
import sys

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF = "mergedd_bn_en.pdf"
STORE = "vectordb"
COLLECTION = "probahini"

# The source is a table of question-and-answer rows, each roughly 300-500
# characters. ~900 holds one or two complete rows; the overlap means a row
# split across a boundary still appears whole in the neighbouring chunk.
#
# Larger would start reintroducing the noise this exists to remove. Smaller
# would cut answers away from their questions.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Chroma writes documents in batches; a few hundred at a time keeps memory
# flat on a small build machine.
BATCH = 200


def main() -> int:
    print(f"loading {PDF}", flush=True)
    pages = PyPDFLoader(PDF).load()
    print(f"  {len(pages)} pages", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        print("ERROR: the PDF produced no text. Is it a scan?", file=sys.stderr)
        return 1

    sizes = [len(c.page_content) for c in chunks]
    print(
        f"  {len(chunks)} chunks, "
        f"min {min(sizes)} / avg {sum(sizes) // len(sizes)} / max {max(sizes)} chars",
        flush=True,
    )

    # Start from nothing. Reusing the directory would leave the old
    # whole-page entries sitting alongside the new chunks, and retrieval would
    # keep finding the very pages this script exists to replace.
    shutil.rmtree(STORE, ignore_errors=True)

    client = chromadb.PersistentClient(STORE)
    collection = client.get_or_create_collection(name=COLLECTION)

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        print(f"  added {start + len(batch)}/{len(chunks)}", flush=True)

    count = collection.count()
    print(f"done: {count} entries in {STORE}/{COLLECTION}", flush=True)

    if count != len(chunks):
        print(
            f"ERROR: expected {len(chunks)} entries, stored {count}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
