-- Core Main durable state model. All timestamps are Unix epoch seconds (REAL).

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE runtimes (
  id TEXT PRIMARY KEY,
  pid INTEGER NOT NULL,
  host TEXT NOT NULL,
  version TEXT NOT NULL,
  mode TEXT NOT NULL,
  started_at REAL NOT NULL,
  heartbeat_at REAL NOT NULL,
  stopped_at REAL,
  status TEXT NOT NULL CHECK (status IN ('running', 'stopped', 'dead'))
);

CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  root_path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  vcs TEXT NOT NULL DEFAULT 'none',
  git_remote TEXT,
  profile_json TEXT,
  profile_hash TEXT,
  profile_updated_at REAL,
  trusted_config_hash TEXT,
  trusted_at REAL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
  parent_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
  fork_checkpoint_id TEXT,
  intent TEXT,
  summary TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX idx_sessions_project ON sessions(project_id, updated_at DESC);

CREATE TABLE messages (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, seq);
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='seq');
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.seq, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.seq, old.content);
END;
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.seq, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.seq, new.content);
END;

CREATE TABLE tasks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
  parent_task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  status TEXT NOT NULL,
  status_reason TEXT,
  block_reason TEXT,
  mode TEXT,
  decision_json TEXT NOT NULL DEFAULT '{}',
  contract_json TEXT NOT NULL DEFAULT '{}',
  options_json TEXT NOT NULL DEFAULT '{}',
  priority INTEGER NOT NULL DEFAULT 0,
  depth INTEGER NOT NULL DEFAULT 0,
  workspace_id TEXT,
  result_summary TEXT,
  evidence_level TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  recovered_count INTEGER NOT NULL DEFAULT 0,
  cancel_requested_at REAL,
  cancel_reason TEXT,
  pause_requested_at REAL,
  version INTEGER NOT NULL DEFAULT 0,
  idempotency_key TEXT UNIQUE,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  completed_at REAL
);
CREATE INDEX idx_tasks_project_status ON tasks(project_id, status, created_at);
CREATE INDEX idx_tasks_session ON tasks(session_id, created_at);
CREATE INDEX idx_tasks_parent ON tasks(parent_task_id);

CREATE TABLE task_deps (
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  depends_on TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  PRIMARY KEY (task_id, depends_on),
  CHECK (task_id <> depends_on)
);

CREATE TABLE leases (
  resource TEXT PRIMARY KEY,
  owner TEXT,
  token INTEGER NOT NULL DEFAULT 0,
  acquired_at REAL,
  expires_at REAL,
  released_at REAL
);

CREATE TABLE attempts (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  number INTEGER NOT NULL,
  status TEXT NOT NULL,
  runtime_id TEXT,
  fence_token INTEGER NOT NULL,
  workspace_id TEXT,
  workflow TEXT,
  checkpoint_json TEXT NOT NULL DEFAULT '{}',
  base_fingerprint TEXT,
  resumed_from TEXT,
  error_class TEXT,
  error_message TEXT,
  usage_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL NOT NULL,
  heartbeat_at REAL,
  ended_at REAL,
  UNIQUE (task_id, number)
);
CREATE INDEX idx_attempts_status ON attempts(status);
CREATE INDEX idx_attempts_task ON attempts(task_id, number);

CREATE TABLE events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  project_id TEXT,
  session_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  actor TEXT NOT NULL DEFAULT 'runtime',
  data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_events_task ON events(task_id, seq);
CREATE INDEX idx_events_session ON events(session_id, seq);
CREATE INDEX idx_events_kind ON events(kind, seq);

CREATE TABLE workspaces (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('canonical', 'worktree', 'copy')),
  path TEXT NOT NULL,
  branch TEXT,
  base_ref TEXT,
  base_fingerprint TEXT,
  baseline_artifact_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('active', 'released', 'preserved', 'applied', 'discarded', 'deleted', 'orphaned')),
  owner_task_id TEXT,
  owner_attempt_id TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX idx_workspaces_project ON workspaces(project_id, status);

CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
  task_id TEXT,
  attempt_id TEXT,
  kind TEXT NOT NULL,
  name TEXT,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL,
  media_type TEXT NOT NULL DEFAULT 'text/plain',
  state TEXT NOT NULL DEFAULT 'live' CHECK (state IN ('live', 'protected', 'expired')),
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE INDEX idx_artifacts_task ON artifacts(task_id);
CREATE INDEX idx_artifacts_sha ON artifacts(sha256);

CREATE TABLE evidence (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  attempt_id TEXT,
  workspace_id TEXT,
  kind TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pass', 'fail', 'error', 'inconclusive', 'skipped')),
  trust TEXT NOT NULL CHECK (trust IN ('verified', 'observed', 'claimed', 'external')),
  summary TEXT NOT NULL,
  fingerprint TEXT,
  diff_hash TEXT,
  command TEXT,
  exit_code INTEGER,
  tool TEXT,
  provider TEXT,
  model TEXT,
  artifact_id TEXT,
  data_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE INDEX idx_evidence_task ON evidence(task_id, kind, created_at);

CREATE TABLE model_calls (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  node TEXT,
  role TEXT,
  purpose TEXT,
  provider_id TEXT NOT NULL,
  model_key TEXT,
  model_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'error', 'cancelled')),
  error_class TEXT,
  error_message TEXT,
  finish_reason TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cached_tokens INTEGER,
  reasoning_tokens INTEGER,
  cost_usd REAL,
  cost_source TEXT,
  latency_ms INTEGER,
  ttft_ms INTEGER,
  retries INTEGER NOT NULL DEFAULT 0,
  tool_call_count INTEGER NOT NULL DEFAULT 0,
  context_snapshot_id TEXT,
  prompt_version TEXT,
  response_artifact_id TEXT,
  started_at REAL NOT NULL,
  ended_at REAL
);
CREATE INDEX idx_model_calls_task ON model_calls(task_id, started_at);
CREATE INDEX idx_model_calls_model ON model_calls(model_id, started_at);

CREATE TABLE tool_calls (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  model_call_id TEXT,
  tool TEXT NOT NULL,
  capability TEXT,
  side_effect TEXT,
  args_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'error', 'denied', 'cancelled', 'timeout', 'interrupted')),
  error_class TEXT,
  error_message TEXT,
  decision_json TEXT NOT NULL DEFAULT '{}',
  output_summary TEXT,
  artifact_id TEXT,
  started_at REAL NOT NULL,
  ended_at REAL,
  duration_ms INTEGER
);
CREATE INDEX idx_tool_calls_task ON tool_calls(task_id, started_at);
CREATE INDEX idx_tool_calls_status ON tool_calls(status);

CREATE TABLE processes (
  id TEXT PRIMARY KEY,
  runtime_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  tool_call_id TEXT,
  pid INTEGER NOT NULL,
  pgid INTEGER,
  start_ticks TEXT,
  command TEXT NOT NULL,
  cwd TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running', 'exited', 'killed', 'timeout', 'orphaned', 'reaped')),
  exit_code INTEGER,
  started_at REAL NOT NULL,
  ended_at REAL
);
CREATE INDEX idx_processes_status ON processes(status);

CREATE TABLE approvals (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  session_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  tool_call_id TEXT,
  capability TEXT NOT NULL,
  summary TEXT NOT NULL,
  request_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'cancelled')),
  scope TEXT,
  decided_by TEXT,
  reason TEXT,
  created_at REAL NOT NULL,
  decided_at REAL
);
CREATE INDEX idx_approvals_status ON approvals(status, created_at);
CREATE INDEX idx_approvals_task ON approvals(task_id);

CREATE TABLE permission_grants (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  session_id TEXT,
  task_id TEXT,
  capability TEXT NOT NULL,
  pattern TEXT NOT NULL DEFAULT '*',
  decision TEXT NOT NULL CHECK (decision IN ('allow', 'deny')),
  granted_by TEXT NOT NULL,
  reason TEXT,
  created_at REAL NOT NULL,
  expires_at REAL,
  uses_remaining INTEGER,
  revoked_at REAL
);
CREATE INDEX idx_grants_scope ON permission_grants(project_id, capability);

CREATE TABLE context_snapshots (
  id TEXT PRIMARY KEY,
  task_id TEXT,
  attempt_id TEXT,
  node TEXT,
  stage TEXT NOT NULL,
  strategy TEXT NOT NULL,
  strategy_version TEXT NOT NULL,
  cache_key TEXT,
  budget_tokens INTEGER NOT NULL,
  used_tokens INTEGER NOT NULL,
  manifest_json TEXT NOT NULL,
  content_artifact_id TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX idx_ctx_task ON context_snapshots(task_id, created_at);

CREATE TABLE reviews (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  attempt_id TEXT,
  strategy TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  independence TEXT,
  diff_hash TEXT NOT NULL,
  fingerprint TEXT,
  verdict TEXT NOT NULL CHECK (verdict IN ('approve', 'request_changes', 'inconclusive')),
  summary TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX idx_reviews_task ON reviews(task_id, created_at);

CREATE TABLE findings (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  attempt_id TEXT,
  review_id TEXT REFERENCES reviews(id) ON DELETE CASCADE,
  severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info')),
  category TEXT NOT NULL,
  title TEXT NOT NULL,
  file TEXT,
  line INTEGER,
  rationale TEXT NOT NULL,
  evidence TEXT,
  remediation TEXT,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'dismissed')),
  resolution TEXT,
  diff_hash TEXT,
  location_valid INTEGER,
  created_at REAL NOT NULL,
  resolved_at REAL
);
CREATE INDEX idx_findings_task ON findings(task_id, status);

CREATE TABLE memory (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  project_id TEXT,
  session_id TEXT,
  task_id TEXT,
  scope TEXT NOT NULL CHECK (scope IN ('task', 'session', 'project', 'operational', 'global')),
  kind TEXT NOT NULL CHECK (kind IN ('fact', 'hypothesis', 'suggestion', 'preference', 'decision', 'observation', 'procedure')),
  content TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.5,
  source_type TEXT NOT NULL CHECK (source_type IN ('user', 'evidence', 'model', 'tool', 'import', 'external', 'runtime')),
  source_ref TEXT,
  evidence_ids_json TEXT NOT NULL DEFAULT '[]',
  anchors_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'stale', 'invalidated', 'superseded')),
  status_reason TEXT,
  supersedes_id TEXT,
  content_hash TEXT NOT NULL,
  valid_until REAL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  last_used_at REAL,
  use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_memory_scope ON memory(project_id, scope, status);
CREATE VIRTUAL TABLE memory_fts USING fts5(content, tags, content='memory', content_rowid='seq');
CREATE TRIGGER memory_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, content, tags) VALUES (new.seq, new.content, new.tags);
END;
CREATE TRIGGER memory_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tags) VALUES ('delete', old.seq, old.content, old.tags);
END;
CREATE TRIGGER memory_au AFTER UPDATE OF content, tags ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tags) VALUES ('delete', old.seq, old.content, old.tags);
  INSERT INTO memory_fts(rowid, content, tags) VALUES (new.seq, new.content, new.tags);
END;

CREATE TABLE decisions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  session_id TEXT,
  task_id TEXT,
  title TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed', 'accepted', 'superseded', 'rejected')),
  context TEXT NOT NULL,
  decision TEXT NOT NULL,
  alternatives_json TEXT NOT NULL DEFAULT '[]',
  consequences TEXT,
  evidence_ids_json TEXT NOT NULL DEFAULT '[]',
  source TEXT NOT NULL,
  supersedes_id TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX idx_decisions_project ON decisions(project_id, status);

CREATE TABLE failures (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  stage TEXT,
  category TEXT NOT NULL,
  error_class TEXT NOT NULL,
  signature TEXT NOT NULL,
  summary TEXT NOT NULL,
  context_json TEXT NOT NULL DEFAULT '{}',
  hypotheses_json TEXT NOT NULL DEFAULT '[]',
  fixes_json TEXT NOT NULL DEFAULT '[]',
  outcome TEXT NOT NULL DEFAULT 'open' CHECK (outcome IN ('open', 'resolved', 'ignored')),
  root_cause TEXT,
  resolution TEXT,
  resolution_evidence_id TEXT,
  verified INTEGER NOT NULL DEFAULT 0,
  regression_candidate INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  resolved_at REAL
);
CREATE INDEX idx_failures_sig ON failures(signature, created_at);
CREATE INDEX idx_failures_task ON failures(task_id);

CREATE TABLE regression_cases (
  id TEXT PRIMARY KEY,
  failure_id TEXT,
  signature TEXT NOT NULL,
  title TEXT NOT NULL,
  scenario_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
  created_at REAL NOT NULL
);

CREATE TABLE heuristics (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('routing', 'context', 'workflow', 'operational', 'skills')),
  content_json TEXT NOT NULL,
  rationale TEXT NOT NULL,
  source_failure_ids_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL CHECK (status IN ('proposed', 'validated', 'active', 'rejected', 'retired')),
  validation_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE (name, version)
);

CREATE TABLE debug_hypotheses (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  attempt_id TEXT,
  statement TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open', 'confirmed', 'refuted', 'inconclusive')),
  experiments_json TEXT NOT NULL DEFAULT '[]',
  evidence_ids_json TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX idx_hypotheses_task ON debug_hypotheses(task_id);

CREATE TABLE checkpoints (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  session_id TEXT,
  task_id TEXT,
  attempt_id TEXT,
  workspace_id TEXT,
  kind TEXT NOT NULL,
  label TEXT,
  tree_ref TEXT,
  fingerprint TEXT,
  state_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE INDEX idx_checkpoints_session ON checkpoints(session_id, created_at);

CREATE TABLE skill_trust (
  name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('trusted', 'blocked')),
  source TEXT,
  decided_at REAL NOT NULL,
  PRIMARY KEY (name, sha256)
);

CREATE TABLE mcp_state (
  server TEXT PRIMARY KEY,
  config_hash TEXT,
  status TEXT NOT NULL,
  error_class TEXT,
  error_message TEXT,
  protocol_version TEXT,
  server_info_json TEXT,
  tools_json TEXT,
  checked_at REAL NOT NULL
);

CREATE TABLE index_files (
  project_id TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  language TEXT,
  is_test INTEGER NOT NULL DEFAULT 0,
  parse_status TEXT NOT NULL,
  symbol_count INTEGER NOT NULL DEFAULT 0,
  summary TEXT,
  indexed_at REAL NOT NULL,
  PRIMARY KEY (project_id, path)
);
CREATE TABLE index_symbols (
  id INTEGER PRIMARY KEY,
  project_id TEXT NOT NULL,
  path TEXT NOT NULL,
  name TEXT NOT NULL,
  qualname TEXT NOT NULL,
  kind TEXT NOT NULL,
  line INTEGER NOT NULL,
  end_line INTEGER,
  signature TEXT,
  parent TEXT,
  exported INTEGER NOT NULL DEFAULT 1,
  doc TEXT
);
CREATE INDEX idx_symbols_name ON index_symbols(project_id, name);
CREATE INDEX idx_symbols_path ON index_symbols(project_id, path);
CREATE TABLE index_imports (
  project_id TEXT NOT NULL,
  path TEXT NOT NULL,
  module TEXT NOT NULL,
  resolved_path TEXT,
  line INTEGER
);
CREATE INDEX idx_imports_path ON index_imports(project_id, path);
CREATE INDEX idx_imports_resolved ON index_imports(project_id, resolved_path);
CREATE TABLE index_chunks (
  id INTEGER PRIMARY KEY,
  project_id TEXT NOT NULL,
  path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL
);
CREATE INDEX idx_chunks_path ON index_chunks(project_id, path);
CREATE VIRTUAL TABLE index_fts USING fts5(body, idents, path, tokenize = "unicode61 remove_diacritics 2 tokenchars '_'");
CREATE TABLE index_state (
  project_id TEXT PRIMARY KEY,
  indexer_version TEXT NOT NULL,
  files INTEGER NOT NULL DEFAULT 0,
  symbols INTEGER NOT NULL DEFAULT 0,
  head_ref TEXT,
  last_scan_at REAL,
  last_duration_ms INTEGER
);

CREATE TABLE cache_entries (
  namespace TEXT NOT NULL,
  key TEXT NOT NULL,
  value BLOB,
  size INTEGER NOT NULL,
  created_at REAL NOT NULL,
  expires_at REAL,
  hits INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (namespace, key)
);

CREATE TABLE eval_runs (
  id TEXT PRIMARY KEY,
  suite TEXT NOT NULL,
  label TEXT,
  strategy_json TEXT NOT NULL,
  core_version TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  heuristics_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL NOT NULL,
  ended_at REAL
);
CREATE TABLE eval_results (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  scenario TEXT NOT NULL,
  family TEXT NOT NULL,
  status TEXT NOT NULL,
  task_status TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  checks_json TEXT NOT NULL DEFAULT '[]',
  error TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX idx_eval_results_run ON eval_results(run_id);
