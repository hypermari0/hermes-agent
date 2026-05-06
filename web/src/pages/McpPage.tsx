import { useCallback, useEffect, useRef, useState } from "react";
import {
  Plug,
  ShieldCheck,
  ShieldOff,
  RefreshCw,
  ExternalLink,
  RotateCw,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  McpServerInfo,
  McpOAuthFlowState,
  McpOAuthStatus,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Badge } from "@nous-research/ui/ui/components/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";

const POLL_INTERVAL_MS = 1500;

type FlowMap = Record<string, McpOAuthFlowState>;

function statusBadge(status: McpOAuthStatus) {
  switch (status) {
    case "starting":
      return (
        <Badge tone="outline" className="text-amber-500 border-amber-500/40">
          Starting…
        </Badge>
      );
    case "url_ready":
      return (
        <Badge tone="outline" className="text-blue-500 border-blue-500/40">
          Awaiting authorization
        </Badge>
      );
    case "completed":
      return (
        <Badge tone="outline" className="text-emerald-500 border-emerald-500/40">
          Connected
        </Badge>
      );
    case "failed":
      return (
        <Badge tone="outline" className="text-rose-500 border-rose-500/40">
          Failed
        </Badge>
      );
    default:
      return null;
  }
}

export default function McpPage() {
  const { showToast, toast } = useToast();
  const [servers, setServers] = useState<McpServerInfo[] | null>(null);
  const [flows, setFlows] = useState<FlowMap>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [loadingList, setLoadingList] = useState(true);
  const [restarting, setRestarting] = useState(false);

  // Track which servers are being polled so we don't double-poll on rapid clicks.
  const pollingRef = useRef<Set<string>>(new Set());
  // Avoid stale closures in the poll callback.
  const showToastRef = useRef(showToast);
  showToastRef.current = showToast;

  const refreshServers = useCallback(async () => {
    setLoadingList(true);
    try {
      const resp = await api.getMcpServers();
      setServers(resp.servers);
    } catch (err) {
      showToastRef.current(`Failed to load MCP servers: ${err}`, "error");
      setServers([]);
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    refreshServers();
  }, [refreshServers]);

  const pollFlow = useCallback(async (name: string) => {
    if (pollingRef.current.has(name)) return;
    pollingRef.current.add(name);
    try {
      while (pollingRef.current.has(name)) {
        let state: McpOAuthFlowState;
        try {
          state = await api.getMcpOAuthStatus(name);
        } catch (err) {
          showToastRef.current(`Status fetch failed: ${err}`, "error");
          return;
        }
        setFlows((prev) => ({ ...prev, [name]: state }));
        if (state.status === "completed" || state.status === "failed") {
          if (state.status === "completed") {
            // Tokens were just written to disk; refresh the server list so
            // the "logged in" badge updates.
            refreshServers();
          }
          return;
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
    } finally {
      pollingRef.current.delete(name);
    }
  }, [refreshServers]);

  // Resume polling for any server already in a non-terminal state when the
  // page mounts (e.g. user navigated away and came back).
  useEffect(() => {
    if (!servers) return;
    for (const s of servers) {
      if (s.auth !== "oauth") continue;
      api.getMcpOAuthStatus(s.name)
        .then((state) => {
          setFlows((prev) => ({ ...prev, [s.name]: state }));
          if (state.status === "starting" || state.status === "url_ready") {
            pollFlow(s.name);
          }
        })
        .catch(() => {});
    }
    // We deliberately depend only on servers' identity, not flows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers]);

  // Stop all polling on unmount.
  useEffect(() => {
    const ref = pollingRef;
    return () => {
      ref.current.clear();
    };
  }, []);

  const handleConnect = useCallback(async (name: string) => {
    setBusy((b) => ({ ...b, [name]: true }));
    try {
      const state = await api.startMcpOAuth(name);
      setFlows((prev) => ({ ...prev, [name]: state }));
      pollFlow(name);
    } catch (err) {
      showToastRef.current(`Failed to start OAuth: ${err}`, "error");
    } finally {
      setBusy((b) => ({ ...b, [name]: false }));
    }
  }, [pollFlow]);

  const handleReset = useCallback(async (name: string) => {
    pollingRef.current.delete(name);
    try {
      await api.clearMcpOAuth(name);
    } catch {
      // ignore
    }
    setFlows((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
  }, []);

  const handleRestartGateway = useCallback(async () => {
    setRestarting(true);
    try {
      await api.restartGateway();
      showToastRef.current(
        "Gateway restarting — give it a few seconds then refresh.",
        "success",
      );
    } catch (err) {
      showToastRef.current(`Restart failed: ${err}`, "error");
    } finally {
      setRestarting(false);
    }
  }, []);

  const oauthServers = (servers ?? []).filter((s) => s.auth === "oauth");
  const otherServers = (servers ?? []).filter((s) => s.auth !== "oauth");
  const anyCompleted = oauthServers.some(
    (s) => flows[s.name]?.status === "completed",
  );

  return (
    <div className="px-4 sm:px-6 py-4 max-w-4xl mx-auto space-y-4">
      <Toast toast={toast} />

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <Plug className="h-5 w-5 text-muted-foreground" />
                MCP servers
              </CardTitle>
              <CardDescription>
                Drive OAuth handshakes from your browser when the host is
                headless (e.g. Railway). Tokens persist to{" "}
                <code className="text-xs">$HERMES_HOME/mcp-tokens/</code>.
              </CardDescription>
            </div>
            <Button
              size="sm"
              outlined
              onClick={refreshServers}
              disabled={loadingList}
              prefix={loadingList ? <Spinner /> : <RefreshCw />}
            >
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loadingList && servers === null ? (
            <div className="py-8 flex justify-center">
              <Spinner />
            </div>
          ) : oauthServers.length === 0 ? (
            <p className="text-sm text-muted-foreground py-2">
              No OAuth-configured MCP servers in <code>config.yaml</code>.
            </p>
          ) : (
            <ul className="space-y-3">
              {oauthServers.map((s) => {
                const flow = flows[s.name];
                const status: McpOAuthStatus = flow?.status ?? "idle";
                const hasTokens = s.has_tokens === true;
                const isBusy =
                  status === "starting" || status === "url_ready" ||
                  busy[s.name];

                return (
                  <li
                    key={s.name}
                    className="rounded-md border border-border bg-card/50 p-3"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-medium">{s.name}</span>
                          {hasTokens ? (
                            <Badge
                              tone="outline"
                              className="text-emerald-500 border-emerald-500/40"
                            >
                              <ShieldCheck className="h-3 w-3 mr-1" />
                              tokens cached
                            </Badge>
                          ) : (
                            <Badge
                              tone="outline"
                              className="text-amber-500 border-amber-500/40"
                            >
                              <ShieldOff className="h-3 w-3 mr-1" />
                              not authenticated
                            </Badge>
                          )}
                          {statusBadge(status)}
                        </div>
                        {s.url && (
                          <p className="text-xs text-muted-foreground mt-1 truncate">
                            {s.url}
                          </p>
                        )}
                      </div>
                      <div className="flex gap-2 shrink-0">
                        {status === "completed" || status === "failed" ? (
                          <Button
                            size="sm"
                            outlined
                            onClick={() => handleReset(s.name)}
                          >
                            Reset
                          </Button>
                        ) : null}
                        <Button
                          size="sm"
                          onClick={() => handleConnect(s.name)}
                          disabled={isBusy}
                          prefix={isBusy ? <Spinner /> : <Plug />}
                        >
                          {hasTokens ? "Reconnect" : "Connect"}
                        </Button>
                      </div>
                    </div>

                    {status === "url_ready" && flow?.url ? (
                      <div className="mt-3 rounded bg-blue-500/10 border border-blue-500/30 p-3 text-sm">
                        <p className="font-medium mb-1">
                          Open this URL to authorize:
                        </p>
                        <a
                          href={flow.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 break-all"
                        >
                          {flow.url}
                          <ExternalLink className="h-3 w-3 shrink-0" />
                        </a>
                        <p className="text-xs text-muted-foreground mt-2">
                          The provider will redirect back to{" "}
                          <code className="text-xs">/oauth/callback</code> on
                          this dashboard. Status will update automatically.
                        </p>
                      </div>
                    ) : null}

                    {status === "starting" ? (
                      <div className="mt-3 text-xs text-muted-foreground flex items-center gap-2">
                        <Spinner />
                        Contacting{" "}
                        <code className="text-xs">{s.url ?? s.name}</code>…
                      </div>
                    ) : null}

                    {status === "completed" ? (
                      <div className="mt-3 rounded bg-emerald-500/10 border border-emerald-500/30 p-3 text-sm flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                        <span>
                          Connected — discovered{" "}
                          {flow?.tool_count ?? 0} tools. Restart the gateway so
                          it picks up the cached tokens.
                        </span>
                      </div>
                    ) : null}

                    {status === "failed" && flow?.error ? (
                      <div className="mt-3 rounded bg-rose-500/10 border border-rose-500/30 p-3 text-sm flex items-start gap-2">
                        <XCircle className="h-4 w-4 text-rose-500 shrink-0 mt-0.5" />
                        <span className="break-words">{flow.error}</span>
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}

          {anyCompleted ? (
            <div className="mt-4 pt-4 border-t border-border">
              <Button
                onClick={handleRestartGateway}
                disabled={restarting}
                prefix={restarting ? <Spinner /> : <RotateCw />}
              >
                Restart gateway
              </Button>
              <p className="text-xs text-muted-foreground mt-2">
                Once restarted the gateway will reconnect to any server whose
                tokens are now cached, and tools become available to the agent.
              </p>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {otherServers.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Other MCP servers</CardTitle>
            <CardDescription>
              Non-OAuth servers (bearer tokens, stdio commands, etc.) — managed
              via <code className="text-xs">config.yaml</code> directly.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {otherServers.map((s) => (
                <li
                  key={s.name}
                  className="text-sm flex items-center justify-between"
                >
                  <span className="font-medium">{s.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {s.auth ?? (s.command ? "stdio" : "—")}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
