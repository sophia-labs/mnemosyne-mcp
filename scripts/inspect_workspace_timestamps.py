#!/usr/bin/env python3
"""Inspect what the workspace CRDT actually stores for document entries.

Connects to the live workspace and dumps keys/types/values for a few docs.
"""

import asyncio
import os
import pycrdt

from neem.hocuspocus import HocuspocusClient, WorkspaceReader
from neem.hocuspocus.workspace import _entry_to_dict
from neem.mcp.tools.hocuspocus import _normalize_timestamp_to_iso

# Target documents to inspect
TARGET_DOCS = [
    "ground-truth-sophia-labs",
    "toward-noetic-ecosystem",
    "arkwork-ascesis-the-order-of-the-cardinals",
    "noetic-metabolism-v2",
    "arkwork-majesty-gardening",  # GARDENING - has real valuations
]

GRAPH_ID = "default"
# Eschaton's prod user ID
USER_ID = "e9b9590e-e0c1-705c-7a47-6c74fdd54f6e"


async def main():
    base_url = os.environ.get("MNEMOSYNE_BASE_URL", "https://app.mnemosyneai.com")
    token = os.environ.get("MNEMOSYNE_TOKEN") or os.environ.get("SOPHIA_TOKEN")
    if not token:
        from neem.utils.token_storage import validate_token_and_load
        token = validate_token_and_load()
        if not token:
            print("ERROR: No auth token found. Set MNEMOSYNE_TOKEN or SOPHIA_TOKEN.")
            return

    hp = HocuspocusClient(base_url=base_url, token_provider=lambda: token)
    await hp.connect_workspace(GRAPH_ID, user_id=USER_ID)
    ws_channel = hp.get_workspace_channel(GRAPH_ID, user_id=USER_ID)
    if not ws_channel:
        print("ERROR: Could not connect to workspace")
        return

    ws_doc = ws_channel.doc
    reader = WorkspaceReader(ws_doc)

    print(f"Total documents in workspace: {len(list(reader._documents.keys()))}\n")

    for doc_id in TARGET_DOCS:
        print(f"=== {doc_id} ===")

        # Method 1: Raw pycrdt.Map access (old broken path)
        raw_entry = reader._documents.get(doc_id)
        if raw_entry is None:
            print(f"  NOT FOUND in _documents")
            continue

        print(f"  Type: {type(raw_entry).__name__}")
        if isinstance(raw_entry, pycrdt.Map):
            keys = list(raw_entry.keys())
            print(f"  Keys: {keys}")
            for key in keys:
                val = raw_entry.get(key)
                print(f"    {key}: {type(val).__name__} = {repr(val)[:100]}")

            created_raw = raw_entry.get("createdAt")
            print(f"\n  raw .get('createdAt'): {type(created_raw).__name__} = {repr(created_raw)}")
            normalized = _normalize_timestamp_to_iso(created_raw)
            print(f"  normalized: {normalized}")

        # Method 2: get_document (dict conversion)
        doc_meta = reader.get_document(doc_id)
        if doc_meta:
            print(f"\n  get_document() keys: {list(doc_meta.keys())}")
            created_dict = doc_meta.get("createdAt") or doc_meta.get("created_at")
            print(f"  dict createdAt: {type(created_dict).__name__ if created_dict else 'None'} = {repr(created_dict)}")
            normalized2 = _normalize_timestamp_to_iso(created_dict)
            print(f"  dict normalized: {normalized2}")

        # Method 3: _entry_to_dict directly
        as_dict = _entry_to_dict(raw_entry)
        if as_dict:
            created_e2d = as_dict.get("createdAt") or as_dict.get("created_at")
            print(f"\n  _entry_to_dict createdAt: {repr(created_e2d)}")

        print()

    await hp.close()


if __name__ == "__main__":
    asyncio.run(main())
