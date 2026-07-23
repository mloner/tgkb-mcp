# tgkb — Telegram Knowledge Base

**Turn the Telegram channels and groups you follow into a private, searchable knowledge base you can query from any MCP-compatible AI assistant in plain language — with every answer cited back to the original message.**

`tgkb` authenticates as your Telegram account, exports message history into a local SQLite database with a **Russian-lemmatized full-text index**, and exposes it through a **read-only [MCP](https://modelcontextprotocol.io) server**. You ask a question in natural language; the assistant searches your archive and answers with links to the real messages — instead of you scrolling through years of chat history.

Because it speaks the standard Model Context Protocol over stdio, it works with **any MCP client** — Claude Code, Claude Desktop, Cursor, and other MCP-capable AI tools. Claude Code is used for the examples below; the setup is the same everywhere, only the config-file location differs.

It is built as two processes that share **one** SQLite file and never talk to each other directly:

- **Writer** (`tgkb` CLI) — authenticates with Telegram, exports history, and keeps a realtime sync running. It is the ONLY thing that writes.
- **Reader** (`tgkb-mcp` MCP server) — a read-only MCP server your AI client launches over stdio. It never writes, never needs an AI key, and exposes three tools: `server_status`, `search_messages`, `read_query`.

Everything local lives under `./.tgkb-data/` (gitignored): the encrypted session, the config, and the SQLite database.

---

## Features

- **Natural-language Q&A over your chats** — ask your AI assistant in plain language; it composes the search/SQL calls for you and answers from your own message archive.
- **Works with any MCP client** — standard Model Context Protocol over stdio, so any MCP-capable AI tool (Claude Code, Claude Desktop, Cursor, …) can use it. No lock-in to one assistant.
- **Cited answers** — every hit comes back with sender, date, chat, and a `t.me` deep link (or `null` for private chats) plus surrounding context messages, so you can verify the source.
- **Russian morphological search** — full-text search over a custom `ru_lemma` tokenizer: an inflected query like `договор` matches `договоры` / `договором` / `договорами`. No embeddings, no AI key.
- **Guarded read-only SQL** — a `read_query` tool for precise lookups (counts, filters, date ranges, thread reconstruction). Only a single `SELECT` / `WITH … SELECT` / `EXPLAIN` runs; anything the AI does through the reader is read-only by construction (writes rejected at the boundary *and* the connection opened read-only), so it can never modify your store or act in Telegram.
- **Local & private** — your archive is a plain SQLite file on your machine. Nothing is uploaded; the reader needs no cloud service and no API key.
- **Full history + edits + deletions** — bulk export via Telegram's takeout, then live ingest of new and edited messages. Deleted messages are tombstoned (kept out of results, not silently lost).
- **Account-safe ingestion** — a proactive request-rate throttle plus FLOOD_WAIT compliance keeps the export/sync within Telegram's limits.
- **Secrets encrypted at rest** — the session and `api_hash` are stored passphrase-encrypted under `./.tgkb-data/`; nothing sensitive ever lands in the repo.

---

## Prerequisites

- **Python ≥ 3.12** and **[uv](https://docs.astral.sh/uv/)** (all commands run through `uv run`, which manages the venv for you).
- **Telegram API credentials** — create an app at <https://my.telegram.org> → *API development tools* to get an **api_id** and **api_hash**. You also need the **phone number** of the account.

Install dependencies (the first `uv run …` does this automatically, or force it):

```bash
uv sync
```

---

## Quick start

The full setup, end to end. Each command is explained in detail in the numbered steps below — start here, drop down to a step if you need more.

```bash
# 1. Telegram credentials — copy the template and fill in your api_id / api_hash / phone
cp .env.example .env
$EDITOR .env

# 2. Authenticate once (opens a login-code prompt; sets an at-rest passphrase)
uv run tgkb auth

# 3. Declare which chats to track — copy the template and list your chats
cp config.json.example .tgkb-data/config.json
$EDITOR .tgkb-data/config.json
uv run tgkb config-check          # validate offline before exporting

# 4. Pull history, then keep it synced live
uv run tgkb export                # one-time bulk backfill
uv run tgkb sync                  # catch-up + realtime; runs until Ctrl-C

# 5. Register the reader with your AI client (see Step 5 for the config block)
```

You configure **three** things in total: `.env` (credentials, step 1), `.tgkb-data/config.json` (tracked chats, step 3), and your MCP client's config (the reader registration, step 5). Everything else is automatic.

> On Windows PowerShell, use `Copy-Item .env.example .env` and `notepad .env` instead of `cp` / `$EDITOR`.

---

## Step 1 — Provide your Telegram credentials

The writer reads three values: `TG_API_ID`, `TG_API_HASH`, `TG_PHONE`. Supply them via a local `.env` (recommended) or real environment variables. Anything missing is prompted for interactively (secrets use no-echo input).

Copy the template (`.env.example`) to `.env` in the project root — `.env` is gitignored, so your real `api_hash` is never committed:

```bash
cp .env.example .env
```

Then edit `.env` to fill in your values:

```dotenv
TG_API_ID=1234567
TG_API_HASH=0123456789abcdef0123456789abcdef
TG_PHONE=+15551234567
```

These are only needed for the **first** `tgkb auth`. After that they are persisted encrypted in the session, and later runs read them from there — no env vars required.

## Step 2 — Authenticate (one time)

```bash
uv run tgkb auth
```

- Enter the **login code** Telegram sends you (and your **2FA password** if the account has one).
- You will be asked to set a **passphrase**. This encrypts the saved session + `api_hash` at rest under `./.tgkb-data/`. You will re-enter this passphrase on every later `export`/`sync`. **There is no recovery** — if you forget it, delete `./.tgkb-data/` and re-run `tgkb auth`.

## Step 3 — Declare which chats to track

Create `./.tgkb-data/config.json` (copy `config.json.example` as a starting point). It is a JSON object with a `chats` array; each entry needs **exactly one** of `username` or `id`:

```json
{
  "chats": [
    { "username": "durov", "title": "Pavel Durov's channel" },
    { "id": -1001234567890, "title": "Team group" }
  ],
  "throttle": { "rate_per_second": 1.0 }
}
```

Field reference (JSON has no comments, so the fields are documented here):

| Field | Where | Required | Meaning |
|---|---|---|---|
| `chats` | root | yes | Array of chat entries; must have at least one. |
| `chats[].username` | entry | one of username/id | Public `@name`, **without** the leading `@` (a leading `@` is stripped). |
| `chats[].id` | entry | one of username/id | Numeric peer id, **unquoted** whole number (e.g. `-1001234567890`). |
| `chats[].title` | entry | no | Free-text label for your own readability; ignored by resolution. |
| `throttle.rate_per_second` | root | no | Proactive request-rate cap (Telegram flood-wait safety). Omit the whole `throttle` object to use the built-in default; `0` disables the proactive limit. |

Setting **both** `username` and `id` on one entry, or **neither**, is an error for that entry only — the rest of the file still loads.

Validate it offline (no network) before exporting:

```bash
uv run tgkb config-check
```

Bad entries are reported and skipped; good entries still run.

## Step 4 — Export history, then keep it synced

```bash
uv run tgkb export     # bulk backfill of each tracked chat's text history (via a takeout session)
uv run tgkb sync       # catch-up backfill + realtime listener; runs until you stop it (Ctrl-C)
```

- `export` is the one-time bulk pull. Run it once per newly-tracked chat.
- `sync` catches up anything missed, then stays running to ingest new & edited messages live. Add `--log-level DEBUG` to trace every event (type + chat_id + message_id — never message text).

After this, `./.tgkb-data/tgkb.sqlite3` holds your searchable knowledge base (with a Russian-lemmatized full-text index).

## Step 5 — Query it from your AI assistant

Register the reader with your MCP client by adding this entry to its MCP config.
The config file's location depends on the client:

- **Claude Code** — `.mcp.json` in the project root, or `~/.claude.json`.
- **Claude Desktop** — `claude_desktop_config.json` (Settings → Developer → Edit Config).
- **Cursor** — `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json`.
- **Other MCP clients** — see the client's docs; they all take the same `mcpServers` shape.

Merge the `tgkb` block into your existing `mcpServers` object:

```json
{
  "mcpServers": {
    "tgkb": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/tgkb-mcp", "run", "tgkb-mcp"]
    }
  }
}
```

Replace `/absolute/path/to/tgkb-mcp` with the **absolute path to your
clone of this repo**. `uv --directory <path>` makes `uv` resolve this project's
environment no matter which working directory your MCP client launches from — so
you don't have to start the client from inside the repo. On Windows, use an
absolute path with escaped backslashes, e.g.
`"C:\\Users\\you\\projects\\tgkb-mcp"`.

If you use Claude Code, its CLI can write the same entry for you (run it from
inside the repo so the path is captured automatically):

```bash
claude mcp add tgkb -- uv --directory "$(pwd)" run tgkb-mcp
```

Approve the MCP server when your client prompts. (If the project venv is already on
`PATH`, `"command": "tgkb-mcp"` with no `args` also works.)

You then have three tools available:

| Tool | What it does |
|------|--------------|
| `server_status` | Health probe: message count, whether search is available, read-only flag. |
| `search_messages` | Ranked Russian full-text search. Returns cited hits — text, sender, date, chat, a `t.me` link (or `null` for private chats), and surrounding context messages. Inflected queries work (`договор` finds `договоры`/`договором`). |
| `read_query` | **Guarded** read-only SQL for precise structured lookups. Runs only a single `SELECT` / `WITH … SELECT` / `EXPLAIN`; any write, DDL, multi-statement, or comment-obfuscated write is rejected at the boundary. Returns rows as column-keyed dicts (`[]` when nothing matches). |

Just ask your assistant in natural language — it composes the tool calls. Examples:

- *"How many messages are in the knowledge base?"* → `read_query("SELECT count(*) FROM messages")` or `server_status`.
- *"Find messages about the budget meeting and cite them."* → `search_messages(query="бюджет совещание")`.
- *"Show the last 10 messages in chat 1, newest first."* → `read_query("SELECT message_id, sender_name, text FROM messages WHERE chat_id = 1 ORDER BY message_id DESC", limit=10)`.

### The data you can query (`read_query` schema)

- **`messages`** — `chat_id`, `message_id`, `sender_id`, `sender_name`, `text`, `date_utc`, `reply_to_msg_id`, `edit_date`, `deleted` (`1` = tombstoned; `search_messages` already excludes these).
- **`chats`** — `chat_id`, `title`, `username`, `is_public`.
- **`sync_state`** — export/sync bookkeeping (`last_id` per chat).

---

## Troubleshooting

- **`uv run tgkb doctor`** — offline self-checks (no network), a fast way to confirm the install is sane.
- **"no knowledge base found …"** from the reader — you haven't run `tgkb export` / `tgkb sync` yet; the reader never creates the DB.
- **"search index is stale (tokenizer drift)"** — the persisted tokenizer fingerprint no longer matches the running `ru_lemma`, so the reader **refuses to search** rather than return wrong results. It cannot repair the index (that is the writer's job) — run `tgkb sync` / `tgkb export` to rebuild.
- **Forgot the passphrase** — delete `./.tgkb-data/` and re-run `tgkb auth`.
- **Check the reader starts** — `uv run tgkb-mcp` launches the stdio MCP server directly (Ctrl-C to stop). It speaks MCP JSON-RPC on stdout with no human-readable output; diagnostics go to stderr. It is meant to be driven by an MCP client, not run by hand — but a clean start confirms the wiring.

---

## Limitations

- **Text only.** Only message text is indexed and searchable. Media, files, stickers, polls, and reactions are not stored.
- **Tuned for Russian.** Full-text search uses a Russian lemmatizer (`ru_lemma`). Other languages are still searchable, but only English-ish exact/loose matching — inflection-aware matching is Russian-specific.
- **Not live Telegram.** You query your local archive, so results are only as fresh as your last `sync`/`export`. A message posted seconds ago is not searchable until it has been ingested.
- **You must have access.** The account you authenticate as must already be a member of / subscribed to each tracked chat; `tgkb` reads what that account can see, it does not join anything for you.

---

## Security & privacy

- **This uses a *user* account, not a bot.** You log in as yourself via Telethon (MTProto). Automating a personal account is **your responsibility** and can carry a ban risk under Telegram's Terms of Service — the account-safe throttle reduces, but does not eliminate, that risk. Use an account you are comfortable automating.
- **Everything stays local.** Messages, the config, and the database never leave your machine. There is no server, no telemetry, no cloud component. The reader has **no** AI/LLM/embedding API key and no code path that reads one — it is deterministic, key-free code.
- **Secrets are encrypted at rest.** Your Telegram session and `api_hash` are stored passphrase-encrypted under `./.tgkb-data/`. That directory (and `.env`) is gitignored — nothing sensitive is committed. Treat the SQLite database as sensitive: it contains the full text of your chats in plaintext.

---

## Development

The project is split into three packages that never cross-import except through the shared SQLite file:

- `tgkb/writer/` — the `tgkb` CLI: auth, export, sync, throttle, doctor.
- `tgkb/reader/` — the `tgkb-mcp` read-only MCP server.
- `tgkb/common/` — shared code: config, DB access, schema, the `ru_lemma` tokenizer, the encrypted secrets store.

Run the test suite (no network, no real Telegram — everything is stubbed):

```bash
uv run pytest -q
```

Contributions are welcome — open an issue or a pull request. Please run the tests before submitting.

---

## How it fits together

```
                (writes)                         (reads, read-only)
  Telegram ──► tgkb (writer CLI) ──► .tgkb-data/tgkb.sqlite3 ◄── tgkb-mcp (reader) ◄── MCP client (any AI)
                auth / export / sync         shared SQLite file        3 MCP tools over stdio
```

- Writer and reader never import each other; they share only the SQLite file.
- The reader adds **no AI/LLM dependency and no API key** — it is deterministic, key-free code.
- All diagnostics go to **stderr**; the reader's stdout carries only the MCP protocol.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute (including commercially).
Just keep the copyright and license notice.
