class VectorDebugger:
    """Utility class to inspect FAISS vector stores."""

    @staticmethod
    def inspect(vectorstore):
        try:
            print("\n--- VECTOR STORE INFO ---")
            print("Total vectors:", vectorstore.index.ntotal)
            print("Embedding dimension:", vectorstore.index.d)
            print("Total documents:", len(vectorstore.docstore._dict))

            print("\n--- SAMPLE DOCUMENTS ---")
            for i, (doc_id, doc) in enumerate(vectorstore.docstore._dict.items()):
                if i >= 3:
                    break
                print(f"\nID: {doc_id}")
                print(doc.page_content[:200])

        except Exception as error:
            print(f"Vector inspection failed: {error}")