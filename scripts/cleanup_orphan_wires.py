#!/usr/bin/env python3
"""One-shot script to find and remove orphan wires from a workspace.

Orphan wires are wires whose source or target document no longer exists
in the workspace documents map.

Usage:
    MNEMOSYNE_FASTAPI_URL=https://api.garden.sophia-labs.com \
        uv run python scripts/cleanup_orphan_wires.py [--apply]

Without --apply, runs in dry-run mode (reports orphans without deleting).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pycrdt

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neem.hocuspocus.client import HocuspocusClient
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, validate_token_and_load


GRAPH_ID = "default"


async def main() -> None:
    apply = "--apply" in sys.argv

    base_url = os.environ.get("MNEMOSYNE_FASTAPI_URL", "http://127.0.0.1:8080")
    dev_user_id = get_dev_user_id()
    internal_secret = get_internal_service_secret()

    print(f"Backend: {base_url}")
    print(f"Graph:   {GRAPH_ID}")
    print(f"Mode:    {'APPLY (will delete orphans)' if apply else 'DRY RUN (report only)'}")
    print()

    client = HocuspocusClient(
        base_url=base_url,
        token_provider=validate_token_and_load,
        dev_user_id=dev_user_id,
        internal_service_secret=internal_secret,
    )

    try:
        # Connect to workspace
        print("Connecting to workspace...")
        await client.connect_workspace(GRAPH_ID)
        channel = client.get_workspace_channel(GRAPH_ID)
        if channel is None:
            print("ERROR: Could not connect to workspace")
            return

        doc = channel.doc

        # Read documents map
        documents_map: pycrdt.Map = doc.get("documents", type=pycrdt.Map)
        doc_ids = set(documents_map.keys())
        print(f"Documents in workspace: {len(doc_ids)}")

        # Read folders map (some wires might reference folders, though unlikely)
        folders_map: pycrdt.Map = doc.get("folders", type=pycrdt.Map)
        folder_ids = set(folders_map.keys())

        # Read wires map
        wires_map: pycrdt.Map = doc.get("wires", type=pycrdt.Map)
        wire_ids = list(wires_map.keys())
        print(f"Wires in workspace:    {len(wire_ids)}")
        print()

        # Scan for orphans
        orphan_ids: list[str] = []
        orphan_details: list[dict] = []

        for wire_id in wire_ids:
            wire = wires_map.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                orphan_ids.append(wire_id)
                orphan_details.append({"wire_id": wire_id, "reason": "not a Map"})
                continue

            src_doc = wire.get("sourceDocumentId", "")
            tgt_doc = wire.get("targetDocumentId", "")
            predicate = wire.get("predicate", "")
            inverse_of = wire.get("inverseOf", "")

            src_exists = src_doc in doc_ids
            tgt_exists = tgt_doc in doc_ids

            if not src_exists or not tgt_exists:
                reasons = []
                if not src_exists:
                    reasons.append(f"source '{src_doc}' missing")
                if not tgt_exists:
                    reasons.append(f"target '{tgt_doc}' missing")

                orphan_ids.append(wire_id)
                orphan_details.append({
                    "wire_id": wire_id,
                    "source": src_doc,
                    "target": tgt_doc,
                    "predicate": predicate.split("#")[-1] if "#" in predicate else predicate,
                    "inverse_of": inverse_of or None,
                    "reason": "; ".join(reasons),
                })

        print(f"Orphan wires found:    {len(orphan_ids)}")
        print(f"Orphan rate:           {len(orphan_ids)/len(wire_ids)*100:.1f}%")
        print()

        if not orphan_details:
            print("No orphans found. Workspace is clean.")
            return

        # Show sample
        print("Sample orphans (first 20):")
        for detail in orphan_details[:20]:
            pred = detail.get("predicate", "?")
            print(f"  {detail['wire_id']}: {detail.get('source', '?')} --[{pred}]--> {detail.get('target', '?')}")
            print(f"    Reason: {detail['reason']}")

        if len(orphan_details) > 20:
            print(f"  ... and {len(orphan_details) - 20} more")

        print()

        if not apply:
            print("DRY RUN complete. Run with --apply to delete orphans.")
            return

        # Delete orphans in a single transaction
        print(f"Deleting {len(orphan_ids)} orphan wires...")

        orphan_set = set(orphan_ids)

        def _delete_orphans(doc: pycrdt.Doc) -> None:
            wires: pycrdt.Map = doc.get("wires", type=pycrdt.Map)
            deleted = 0
            for wid in list(wires.keys()):
                if wid in orphan_set:
                    del wires[wid]
                    deleted += 1
            print(f"  Deleted {deleted} wires in transaction")

        await client.transact_workspace(GRAPH_ID, _delete_orphans)

        # Verify
        remaining_wires = len(list(wires_map.keys()))
        print(f"\nWires remaining: {remaining_wires}")
        print("Done.")

    finally:
        # Best-effort cleanup
        pass


if __name__ == "__main__":
    asyncio.run(main())
