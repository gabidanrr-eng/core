"""Origin allowlisting, workspace file confinement and target normalization (no browser needed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coremain.browser.origins import (
    FileScope,
    OriginPattern,
    OriginPolicy,
    TargetError,
    file_url_to_path,
    is_loopback_host,
    policy_target,
    resolve_target,
)
from coremain.config.schema import BrowserConfig
from coremain.security.paths import PathViolation

DEFAULTS = BrowserConfig().allowed_origins


def allowed(policy: OriginPolicy, url: str) -> bool:
    return policy.check_url(url).allowed


def test_default_allowlist_admits_only_local_http_and_files() -> None:
    policy = OriginPolicy(DEFAULTS)
    assert allowed(policy, "http://localhost:3000/app")
    assert allowed(policy, "http://127.0.0.1:8000/")
    assert allowed(policy, "http://localhost/")  # '*' port also covers the default port
    assert allowed(policy, "file:///srv/site/index.html")
    assert allowed(policy, "about:blank")
    assert not allowed(policy, "https://localhost:3000/")  # only http is listed by default
    assert not allowed(policy, "http://example.com/")
    assert not allowed(policy, "http://127.0.0.2:8000/")
    assert not allowed(policy, "http://0.0.0.0:8000/")
    assert not allowed(policy, "ftp://localhost/")
    assert not allowed(policy, "javascript:alert(1)")
    assert not allowed(policy, "about:config")
    verdict = policy.check_url("http://example.com:8080/x")
    assert verdict.target == "http://example.com:8080"
    assert "not in browser.allowed_origins" in verdict.reason


def test_pattern_semantics_hosts_ports_schemes_and_paths() -> None:
    policy = OriginPolicy(["https://*.example.com", "http://api.test:8080/v1/*", "[::1]:*"])
    assert allowed(policy, "https://example.com/")  # '*.' also matches the apex, like network rules
    assert allowed(policy, "https://docs.example.com/page")
    assert not allowed(policy, "https://evilexample.com/")
    assert not allowed(policy, "https://docs.example.com:8443/")  # no port in pattern = default port only
    assert not allowed(policy, "http://docs.example.com/")
    assert allowed(policy, "wss://docs.example.com/socket")  # https patterns cover wss
    assert allowed(policy, "http://api.test:8080/v1/users")
    assert not allowed(policy, "http://api.test:8080/admin")
    assert allowed(policy, "http://[::1]:5173/")  # scheme-less pattern: any network scheme
    assert allowed(policy, "https://[::1]:8443/")
    # A CONNECT tunnel only reveals host:port, so path restrictions cannot apply to it.
    assert policy.check_connect("api.test", 8080).allowed
    assert policy.check_connect("docs.example.com", 443).allowed
    assert not policy.check_connect("docs.example.com", 8443).allowed


def test_ws_is_covered_by_http_patterns_and_connect_checks_all_schemes() -> None:
    policy = OriginPolicy(["http://localhost:5173"])
    assert allowed(policy, "ws://localhost:5173/hmr")
    assert policy.check_connect("localhost", 5173).allowed
    assert not policy.check_connect("localhost", 5174).allowed
    assert OriginPolicy(["wss://push.example.com"]).check_connect("push.example.com", 443).allowed


def test_invalid_patterns_are_reported_and_never_allow() -> None:
    policy = OriginPolicy(
        ["ftp://files.example.com", "http://[::1", "http://host:abc", "http://a:b:c", "", "file://remote/x"]
    )
    assert policy.patterns == []
    assert len(policy.invalid) == 6
    assert not allowed(policy, "http://host/")
    with pytest.raises(ValueError):
        OriginPattern.parse("chrome://settings")


def test_star_allows_everything_but_offline_and_deny_domains_still_win() -> None:
    assert allowed(OriginPolicy(["*"]), "https://example.com/")
    assert allowed(OriginPolicy(["*"]), "file:///tmp/x.html")
    offline = OriginPolicy(["*"], offline=True)
    verdict = offline.check_url("https://example.com/")
    assert not verdict.allowed and "offline mode" in verdict.reason
    assert allowed(offline, "http://localhost:3000/")
    assert allowed(offline, "http://127.0.0.1:9/")
    assert allowed(offline, "http://[::1]:8000/")
    assert allowed(offline, "http://app.localhost:3000/")
    assert not offline.check_connect("example.com", 443).allowed
    denied = OriginPolicy(["*"], deny_domains=["*.tracker.test"])
    assert not allowed(denied, "https://cdn.tracker.test/pixel.gif")
    assert "deny_domains" in denied.check_url("https://tracker.test/").reason
    assert allowed(denied, "https://example.com/")


def test_loopback_detection() -> None:
    for host in ("localhost", "LOCALHOST", "app.localhost", "127.0.0.1", "127.8.9.10", "::1", "[::1]"):
        assert is_loopback_host(host), host
    for host in ("0.0.0.0", "10.0.0.1", "example.com", "localhost.example.com", "::ffff:8.8.8.8"):
        assert not is_loopback_host(host), host


def test_file_scope_confines_to_workspace_and_refuses_secrets(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "site").mkdir(parents=True)
    (ws / "site" / "index.html").write_text("<h1>x</h1>")
    (ws / "site" / "my page.html").write_text("<h1>y</h1>")
    (ws / ".env").write_text("TOKEN=abc")
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[remote]")
    (tmp_path / "outside.html").write_text("secret")
    os.symlink(tmp_path / "outside.html", ws / "site" / "link.html")
    scope = FileScope.of(ws, extra_sensitive=["*.private.html"])
    assert scope.check((ws / "site" / "index.html").as_uri()).allowed
    assert scope.check((ws / "site" / "my page.html").as_uri()).allowed  # percent-encoded path
    assert scope.check(ws.as_uri() + "/").allowed  # directory listings inside the workspace
    assert not scope.check((tmp_path / "outside.html").as_uri()).allowed
    assert "outside the workspace" in scope.check((ws / "site" / "link.html").as_uri()).reason
    assert "sensitive" in scope.check((ws / ".env").as_uri()).reason
    assert ".git" in scope.check((ws / ".git" / "config").as_uri()).reason
    assert not scope.check((ws / "site" / "a.private.html").as_uri()).allowed
    assert not scope.check((ws / "site" / ".." / ".." / "outside.html").as_uri()).allowed
    assert not scope.check("file://fileserver/share/x.html").allowed
    assert not FileScope(()).check((ws / "site" / "index.html").as_uri()).allowed


def test_file_url_to_path() -> None:
    assert file_url_to_path("file:///tmp/a%20b.html") == Path("/tmp/a b.html")
    assert file_url_to_path("file://localhost/tmp/x") == Path("/tmp/x")
    assert file_url_to_path("file://server/share") is None
    assert file_url_to_path("http://localhost/x") is None


def test_resolve_target_paths_urls_and_escapes(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "dist").mkdir(parents=True)
    page = ws / "dist" / "index.html"
    page.write_text("<p>ok</p>")
    (tmp_path / "outside.html").write_text("no")
    assert resolve_target("dist/index.html", ws) == page.resolve().as_uri()
    assert resolve_target("./dist/index.html#intro", ws) == page.resolve().as_uri() + "#intro"
    assert resolve_target("dist/index.html?x=1", ws) == page.resolve().as_uri() + "?x=1"
    assert resolve_target(page.as_uri(), ws) == page.as_uri()
    assert resolve_target("http://localhost:3000/a", ws) == "http://localhost:3000/a"
    assert resolve_target("about:blank", ws) == "about:blank"
    with pytest.raises(PathViolation):
        resolve_target("../outside.html", ws)
    with pytest.raises(PathViolation):
        resolve_target(str(tmp_path / "outside.html"), ws)
    with pytest.raises(PathViolation):
        resolve_target((tmp_path / "outside.html").as_uri(), ws)
    with pytest.raises(TargetError, match="does not exist"):
        resolve_target("dist/missing.html", ws)
    with pytest.raises(TargetError) as info:
        resolve_target("localhost:3000", ws)
    assert info.value.hint == "did you mean http://localhost:3000?"
    for bad in ("javascript:alert(1)", "data:text/html,<p>x</p>", "chrome://settings", "http://", "  "):
        with pytest.raises(TargetError):
            resolve_target(bad, ws)
    # Unconfined resolution (user-initiated CLI) accepts paths outside the base directory.
    assert (
        resolve_target("../outside.html", ws, confine=False) == (tmp_path / "outside.html").resolve().as_uri()
    )


def test_policy_targets_are_origins_or_file_urls() -> None:
    assert policy_target("http://localhost:3000/a/b?q=1") == "http://localhost:3000"
    assert policy_target("https://Example.com:443/login") == "https://example.com"
    assert policy_target("http://[::1]:8000/") == "http://[::1]:8000"
    assert policy_target("file:///srv/site/index.html#x") == "file:///srv/site/index.html"
    for local in ("about:blank", "chrome-error://chromewebdata/", "data:text/plain,x", ""):
        assert policy_target(local) == ""
