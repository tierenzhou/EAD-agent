-- Explore UI / gateway control plane: encrypted operator API keys (single-tenant)
-- Default path: ~/.hermes/explore_control.db
-- Override: environment variable EXPLORE_CONTROL_DB
-- WAL is enabled at runtime by ExploreCredentialStore.

CREATE TABLE IF NOT EXISTS credentials (
    purpose TEXT PRIMARY KEY,
    ciphertext BLOB NOT NULL,
    nonce BLOB NOT NULL,
    last4 TEXT NOT NULL,
    updated_at REAL NOT NULL
);
