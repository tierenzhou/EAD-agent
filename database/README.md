# SQLite schema reference (EAD-agent)

This folder holds **SQL snapshots** of the application-managed schemas so teams can diff, review, and apply changes without copying live `.db` files.

## Source of truth

| Database file (typical location) | Canonical definition in code |
|--------------------------------|--------------------------------|
| `~/.hermes/projects/projects.db` (override via store constructor / env) | `projects/store.py` — `_SCHEMA_SQL`, `_SCHEMA_VERSION` |
| `~/.hermes/explore_control.db` (override: `EXPLORE_CONTROL_DB`) | `gateway/control_plane/credential_store.py` — `CREATE TABLE credentials` |

When you change schema in code, **update the matching `.sql` file here** in the same commit so Git history stays aligned.

## Applying on a server (example)

Always **back up** the existing database first, stop the process that holds the DB open, then:

```bash
sqlite3 /path/to/projects.db < database/projects_schema.sql
sqlite3 /path/to/explore_control.db < database/explore_control_schema.sql
```

New installs: tables use `IF NOT EXISTS`; altering existing columns still requires explicit `ALTER` migrations if you add those later.
