# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 320806b1-99d0-4029-99a4-3d8fec2e6276
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Custom transport example — plug in your own source.

Shows how to implement the Transport protocol.
Use this pattern for: SharePoint, Confluence, Notion, Dropbox, internal APIs.
"""
from chunkymonkey import DocumentLoader
from chunkymonkey.transports._protocol import FetchResult


class InMemoryTransport:
    """Transport that serves documents from an in-memory dict.

    Use as a template for SharePoint, Confluence, and other private sources.
    """

    def __init__(self, documents: dict[str, bytes]):
        self._docs = documents

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("mem://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        key = uri[len("mem://"):]
        if key not in self._docs:
            raise FileNotFoundError(f"Document not found: {uri!r}")
        return FetchResult(
            data=self._docs[key],
            detected_mime="text/markdown",
            source_path=key,
        )


if __name__ == "__main__":
    store = {
        "policy.md": b"# Leave Policy\n\nEmployees are entitled to 20 days per year.\n\n## Sick Leave\n\nUnlimited sick leave with documentation.",
        "handbook.md": b"# Employee Handbook\n\n## Code of Conduct\n\nTreat everyone with respect.",
    }

    transport = InMemoryTransport(store)
    loader = DocumentLoader(extra_transports=[transport])

    for uri in ["mem://policy.md", "mem://handbook.md"]:
        chunks = loader.load(uri)
        print(f"\n{uri}: {len(chunks)} chunk(s)")
        for chunk in chunks:
            print(f"  [{chunk.section}] {chunk.content[:60]}...")
