# Pinecone index dimension mismatch: safe operator procedure

The application never deletes or recreates a Pinecone index automatically.

If startup reports that an existing index dimension differs from the configured
embedding model:

1. Stop ingestion and retrieval clients that write to the affected index.
2. Confirm the current index name, dimension, metric, region, namespace usage,
   embedding model, and any downstream consumers in the Pinecone console.
3. Decide whether to restore the previous embedding configuration or migrate to
   a separately named index. Do not change the dimension of an existing index.
4. For a migration, configure a **new index name** and create that index with
   the dimension required by the selected embedding model.
5. Re-embed source documents into the new index using the same model and
   preprocessing configuration that production retrieval will use.
6. Validate representative retrieval queries and metadata before switching the
   application configuration to the new index.
7. Retain the old index and its data until an authorized owner explicitly
   approves a separately managed retention/decommission process.

The application intentionally fails closed on a dimension mismatch. It does
not delete the old index, modify its vectors, or automatically switch names.
