"""MongoDB Atlas Vector Search index definitions and verification.

December 2025: Index specifications for SOTA RAG system.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Required indexes for December 2025 SOTA RAG
REQUIRED_INDEXES = {
    "chunks": [
        {
            "name": "vector_index_full",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,  # Voyage 3.5 dimension
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "level"},
                    {"type": "filter", "path": "document_id"},
                ]
            },
        },
        {
            "name": "vector_index_binary",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding_binary",
                        "numDimensions": 1024,
                        "similarity": "cosine",
                        "quantization": {"type": "binary"},  # 32x compression
                    }
                ]
            },
        },
        {
            "name": "text_search_index",
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "content": {"type": "string", "analyzer": "lucene.standard"},
                    }
                }
            },
        },
    ],
    "entities": [
        {
            "name": "entity_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "type"},
                    {"type": "filter", "path": "community_id"},
                ]
            },
        },
    ],
    "communities": [
        {
            "name": "community_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "level"},
                ]
            },
        },
    ],
    "documents": [
        {
            "name": "document_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "page_embeddings",
                        "numDimensions": 128,  # ColQwen2 dimension
                        "similarity": "cosine",
                    },
                ]
            },
        },
    ],
}


def get_index_definitions(collection: str | None = None) -> dict[str, list[dict]]:
    """Get index definitions for one or all collections.

    Args:
        collection: Specific collection name, or None for all

    Returns:
        Dict of collection -> index definitions
    """
    if collection:
        return {collection: REQUIRED_INDEXES.get(collection, [])}
    return REQUIRED_INDEXES


def generate_atlas_cli_commands() -> list[str]:
    """Generate MongoDB Atlas CLI commands for creating indexes.

    Returns:
        List of atlas CLI commands
    """
    commands = []
    for collection, indexes in REQUIRED_INDEXES.items():
        for index in indexes:
            cmd = f"""
# Create {index['name']} on {collection}
atlas clusters search indexes create \\
  --clusterName <CLUSTER_NAME> \\
  --db <DATABASE_NAME> \\
  --collection {collection} \\
  --name {index['name']} \\
  --type {index['type']} \\
  --file index_{collection}_{index['name']}.json
"""
            commands.append(cmd.strip())
    return commands


async def verify_indexes(mongodb_client: Any) -> dict[str, list[dict[str, Any]]]:
    """Verify that required indexes exist.

    December 2025: Verify all required indexes are in place.

    Args:
        mongodb_client: MongoDB client instance

    Returns:
        Dict with 'missing', 'present', and 'unknown' index lists
    """
    results = {
        "present": [],
        "missing": [],
        "unknown": [],
    }

    db = mongodb_client.db

    for collection, required in REQUIRED_INDEXES.items():
        # Get existing indexes
        try:
            # Note: This requires admin access to list search indexes
            existing_indexes = await db.command({
                "listSearchIndexes": collection
            })
            existing_names = {idx["name"] for idx in existing_indexes.get("cursor", {}).get("firstBatch", [])}
        except Exception as e:
            logger.warning(f"Could not list indexes for {collection}: {e}")
            existing_names = set()

        # Check each required index
        for index_def in required:
            index_info = {
                "collection": collection,
                "name": index_def["name"],
                "type": index_def["type"],
            }

            if index_def["name"] in existing_names:
                results["present"].append(index_info)
            else:
                results["missing"].append(index_info)

    logger.info(
        f"Index verification: {len(results['present'])} present, "
        f"{len(results['missing'])} missing"
    )
    return results


def print_index_summary() -> str:
    """Print human-readable summary of required indexes.

    Returns:
        Formatted summary string
    """
    lines = ["# December 2025 SOTA RAG - Required MongoDB Indexes", ""]

    for collection, indexes in REQUIRED_INDEXES.items():
        lines.append(f"## Collection: {collection}")
        for idx in indexes:
            lines.append(f"- **{idx['name']}** ({idx['type']})")
            if idx["type"] == "vectorSearch":
                fields = idx["definition"]["fields"]
                for field in fields:
                    if field["type"] == "vector":
                        dims = field.get("numDimensions", "?")
                        quant = field.get("quantization", {}).get("type", "none")
                        lines.append(f"  - Vector: {field['path']} ({dims}D, {quant})")
                    elif field["type"] == "filter":
                        lines.append(f"  - Filter: {field['path']}")
        lines.append("")

    return "\n".join(lines)


# Quick verification helper
def list_required_index_names() -> list[str]:
    """List all required index names.

    Returns:
        List of index names
    """
    names = []
    for indexes in REQUIRED_INDEXES.values():
        for idx in indexes:
            names.append(idx["name"])
    return names


if __name__ == "__main__":
    print(print_index_summary())
    print("\nRequired indexes:", list_required_index_names())
