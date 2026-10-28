# TokenGuard

<p align="left">
  <b>Zero-trust local circuit breaker and real-time telemetry proxy for autonomous LLM agents.</b>
</p>

<p align="left">
  <a href="https://github.com/xerj-org/xerj"><img src="https://img.shields.io/badge/Indexed_in-XERJ_Corpus_Hub-007EC6?style=flat-square&logo=github&logoColor=white" alt="XERJ Hub"></a>
  <img src="https://img.shields.io/badge/Status-Active_%26_Guarding-10b981?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10_%7C_3.11_%7C_3.12-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Providers-OpenAI_%7C_Claude_%7C_DeepSeek-8b5cf6?style=flat-square" alt="Providers">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square" alt="License"></a>
</p>

---

##  Table of contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
  - [High-Level Architecture Diagram](#high-level-architecture-diagram)
  - [Request Lifecycle Sequence](#request-lifecycle-sequence)
- [Visual Showcase](#-visual-showcase)
- [Screenshot Guide & Assets Checklist](#-screenshot-guide--assets-checklist)
- [Core Features & Technical Deep Dive](#-core-features--technical-deep-dive)
  - [1. Infinite Loop & Tool Ping-Pong Circuit Breaker](#1-infinite-loop--tool-ping-pong-circuit-breaker)
  - [2. Velocity & Burn-Rate Circuit Breaker](#2-velocity--burn-rate-circuit-breaker)
  - [3. Context Ballooning Guard](#3-context-ballooning-guard)
  - [4. Observation Mode (`--dry-run` / Shadow Mode)](#4-observation-mode---dry-run--shadow-mode)
  - [5. Saved Tokens & Cost Telemetry](#5-saved-tokens--cost-telemetry)
  - [6. True Zero-Latency Streaming (SSE Proxying)](#6-true-zero-latency-streaming-sse-proxying)
  - [7. Budget Limiter & Avoided Cost Ledger](#7-budget-limiter--avoided-cost-ledger)
  - [8. Dynamic Real-Time Pricing Synchronizer](#8-dynamic-real-time-pricing-synchronizer)
  - [9. Multi-Format Telemetry Exporter](#9-multi-format-telemetry-exporter)
  - [10. Zero-Friction Python SDK](#10-zero-friction-python-sdk)
  - [11. High-End Developer CLI & Web Dashboard](#11-high-end-developer-cli--web-dashboard)
- [Installation & Quickstart](#-installation--quickstart)
- [Complete CLI Command Reference](#-complete-cli-command-reference)
  - [1. `tokenguard start`](#1-tokenguard-start)
  - [2. `tokenguard stop`](#2-tokenguard-stop)
  - [3. `tokenguard status`](#3-tokenguard-status)
  - [4. `tokenguard run`](#4-tokenguard-run)
  - [5. `tokenguard env` (Terminal Environment Injector)](#5-tokenguard-env-terminal-environment-injector)
  - [6. `tokenguard simulate` (Loop & Tool Cycle Simulator)](#6-tokenguard-simulate-loop--tool-cycle-simulator)
  - [7. `tokenguard export`](#7-tokenguard-export)
  - [8. `tokenguard prices`](#8-tokenguard-prices)
  - [9. Reactive CLI Control (`kill`, `resume`, `profile`, `clear`, `logs`)](#9-reactive-cli-control-kill-resume-profile-clear-logs)
  - [10. `tokenguard quickstart`](#10-tokenguard-quickstart)
  - [Interactive TUI Navigation](#interactive-tui-navigation)
- [Python SDK Guide & Code Recipes](#-python-sdk-guide--code-recipes)
  - [Function Decorator (`@protect`)](#function-decorator-protect)
  - [Context Manager (`with guard():`)](#context-manager-with-guard)
  - [Framework Integrations (LangChain, CrewAI, AutoGen, OpenAI)](#framework-integrations)
- [REST API Reference](#-rest-api-reference)
  - [Circuit Breaker 429 Error Format & Custom Headers](#circuit-breaker-429-error-format--custom-headers)
- [Model Pricing Matrix](#-model-pricing-matrix)
- [Automated Testing & CI](#-automated-testing--ci)
- [Contributing & License](#-contributing--license)

---

## Overview

Autonomous LLM agents (e.g. AutoGPT, CrewAI, LangGraph, Cursor background tasks, reasoning loops) frequently get trapped in **recursive prompt loops**, repeating identical tool calls or queries hundreds of times in seconds. Without guardrails, this can consume hundreds or thousands of dollars in API credits within minutes.

**TokenGuard** is an open-source, ultra-low latency **local proxy and circuit breaker sidecar** designed to protect your API budget and monitor LLM traffic with zero code friction.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 YOUR APPLICATION / AGENT                │
                  │   (OpenAI SDK / Anthropic SDK / LangChain / CLI / SDK)   │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │                   TOKENGUARD PROXY                      │
                  │                 http://127.0.0.1:8080                   │
                  │                                                         │
                  │  ● SHA-256 Prompt Hashing      ● Real-Time Budget Cap   │
                  │  ● Sliding Window Loop Check   ● Avoided Cost Ledger    │
                  │  ● Tool Ping-Pong Detection    ● Observation Shadow Mode│
                  │  ● Live OpenRouter Pricing     ● SQLite WAL Telemetry   │
                  └────────────┬───────────────────────────────┬────────────┘
                               │                               │
                      [Allowed Requests]               [Blocked Violations]
                               │                               │
                               ▼                               ▼
                 ┌───────────────────────────┐   ┌───────────────────────────┐
                 │    UPSTREAM LLM PROVIDERS │   │   INSTANT 429 SHUTOFF     │
                 │   OpenAI, Claude, DeepSeek│   │  No upstream token cost   │
                 │   Gemini, Qwen, Mistral   │   │  Avoided cost accounted   │
                 └───────────────────────────┘   └───────────────────────────┘
```

### Why TokenGuard?

- **Zero-latency in-memory guardrails**: Sub-millisecond pre-flight checks using sliding-window prompt hashing and sliding expenditure counters.
- **Multi-step tool cycle detection**: Detects identical tool loops ($A \to A \to A$), alternating tool ping-pong cycles ($A \to B \to A \to B$), and 3-step loops ($A \to B \to C \to A \to B \to C$) while automatically stripping dynamic nonces and UUIDs.
- **Observation shadow mode (`--dry-run`)**: Safely test circuit breaker trip rules against live production agent traffic without cutting connections.
- **Saved tokens & USD cost telemetry**: Automated calculation of intercepted token & USD savings delivered directly via custom HTTP response headers (`X-TokenGuard-*`) and JSON bodies.
- **Terminal environment injector (`tokenguard env`)**: Instantly route OpenAI, Anthropic, Gemini, and DeepSeek SDKs through TokenGuard with `eval $(tokenguard env)`.
- **Drop-in compatibility**: Universal OpenAI-compatible `/v1/chat/completions` endpoint, native Anthropic `/v1/messages` endpoint, and Claude OpenAI adapter.
- **Never breaks future prompts**: Loop rejections never poison future requests; sending a distinct prompt immediately succeeds without false positives.
- **Live dynamic pricing**: Syncs token rates directly with OpenRouter's live registry with offline caching and instant failover.
- **Enterprise-grade telemetry**: Query telemetry in non-blocking SQLite WAL mode with indexed `session_id` and export to **HAR 1.2** (Chrome DevTools / Postman compatible with custom `_tokenCost` annotations), **JSONL**, **JSON**, and **CSV**.
- **Dual developer experience**: Includes a sleek dark-mode Web Dashboard with live Server-Sent Events (SSE) and a minimal, keyboard-driven Stripe-styled Terminal TUI.

---

## System architecture

### High-level architecture diagram

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Agent Layer"]
        CLI["CLI Wrapper (tokenguard run)"]
        SDK["Python SDK (@protect / with guard)"]
        AgentFramework["Agent Frameworks (CrewAI, LangChain, AutoGen)"]
        DirectClient["Direct HTTP / OpenAI SDK Clients"]
    end

    subgraph ProxyCore["TokenGuard Core (127.0.0.1:8080)"]
        Router["Universal Router & Protocol Adapter"]
        
        subgraph GuardEngine["Autonomous Circuit Breaker Engine"]
            Hasher["Deterministic SHA-256 Prompt Hasher"]
            LoopDetector["Sliding-Window Loop Detector (TTL Window)"]
            BudgetGuard["Sliding Hourly & Daily Budget Guard"]
            KillSwitch["Emergency Hardware Kill Switch"]
            AvoidedCost["Avoided Cost & Savings Ledger"]
        end

        subgraph PricingSync["Dynamic Pricing Engine"]
            LiveSync["OpenRouter Live Registry Synchronizer"]
            TTLStorage["Local Cache (~/.tokenguard/prices_cache.json)"]
            StaticFallback["Baseline Registry (prices.json)"]
        end

        subgraph StorageTelemetry["Telemetry & Persistence"]
            SQLiteWAL[("SQLite WAL Database (tokenguard.db)")]
            SSEBroadcaster["Live SSE Broadcaster (/api/stream)"]
            ExportEngine["Multi-Format Exporter (HAR 1.2, JSONL, CSV, JSON)"]
        end
    end

    subgraph UpstreamLLMs["Upstream LLM Providers"]
        OpenAI["OpenAI (GPT-4o, o1, o3-mini)"]
        Anthropic["Anthropic (Claude 3.5 Sonnet, 3.7)"]
        DeepSeek["DeepSeek (deepseek-chat, reasoner)"]
        Gemini["Google Gemini (2.0 Flash, 1.5 Pro)"]
        Qwen["Alibaba Qwen / DashScope (Qwen-Max, QwQ)"]
        Groq["Groq (Llama-3.3, Gemma2)"]
        Mistral["Mistral AI (Codestral, Large)"]
        OpenRouter["OpenRouter (400+ Unified Models)"]
    end

    subgraph UILayer["Developer Interfaces"]
        WebDashboard["Next.js Web Dashboard (Dark Mode & Charts)"]
        TerminalTUI["Rich Terminal TUI (Interactive Control Center)"]
        ExternalTools["Chrome DevTools / Postman (via HAR 1.2)"]
    end

    %% Client to Proxy
    CLI --> Router
    SDK --> Router
    AgentFramework --> Router
    DirectClient --> Router

    %% Routing to Breaker
    Router --> Hasher
    Hasher --> LoopDetector
    LoopDetector --> BudgetGuard
    BudgetGuard --> KillSwitch

    %% Pricing to Breaker
    LiveSync --> TTLStorage
    TTLStorage --> StaticFallback
    StaticFallback --> PricingSync
    PricingSync --> GuardEngine

    %% Breaker Deciding
    KillSwitch -- "VIOLATION (429)" --> AvoidedCost
    AvoidedCost --> SQLiteWAL
    AvoidedCost --> SSEBroadcaster

    KillSwitch -- "ALLOWED (200)" --> UpstreamLLMs
    UpstreamLLMs --> SQLiteWAL
    UpstreamLLMs --> SSEBroadcaster

    %% Persistence to Interfaces
    SQLiteWAL --> ExportEngine
    ExportEngine --> ExternalTools
    SQLiteWAL --> WebDashboard
    SSEBroadcaster --> WebDashboard
    SQLiteWAL --> TerminalTUI
```

---

### Request lifecycle sequence

```mermaid
sequenceDiagram
    autonumber
    actor Developer as Developer / Agent
    participant TG as TokenGuard Proxy
    participant Breaker as Circuit Breaker Engine
    participant DB as SQLite WAL Database
    participant Upstream as Upstream Provider (OpenAI/Claude)
    participant UI as Web Dashboard & SSE

    Developer->>TG: POST /v1/chat/completions (model, messages, stream)
    TG->>Breaker: Check pre-flight guards (Prompt Hash, Loop History, Budget)
    
    alt Loop Detected OR Budget Exceeded OR Kill Switch Active
        Breaker-->>TG: Trip Circuit Breaker (LoopDetectedError / BudgetExceededError)
        TG->>DB: Log request as BLOCKED (cost=0, latency=0, avoided_cost)
        TG->>UI: Broadcast SSE 'request_logged' & 'state_change'
        TG-->>Developer: HTTP 429 Too Many Requests (JSON Circuit Breaker Error)
    else Pre-flight Passed (Normal Traffic)
        Breaker->>Breaker: Record prompt hash in rolling TTL history
        TG->>Upstream: Forward request with upstream credentials
        Upstream-->>TG: HTTP 200 Stream / Response (Tokens + Usage)
        TG->>Breaker: Record actual tokens & calculate exact USD cost
        TG->>DB: Log request as SUCCESS (tokens, cost_usd, latency_ms)
        TG->>UI: Broadcast SSE 'request_logged'
        TG-->>Developer: Stream response transparently back to client
    end
```

---

## Visual showcase

| Screenshot | Description & visual capabilities |
| :--- | :--- |
| ![Dashboard Overview](assets/images/tokenguard-loop-detection.png) | Automatic detection and termination of alternating tool loops (A→B→A→B) before calls reach upstream LLM providers, reporting avoided cost in real time. |
| ![Live Request Table](assets/images/tokenguard-web-dashboard.png) | Live request inspection, kill-switch control, budget tracking, and direct configuration snippets for Python, TypeScript, and shell environments. |
| ![Developer Terminal TUI](assets/images/tokenguard-cli-profiles.png) | Instant provider key detection across OpenAI, Anthropic, Gemini and DeepSeek, with pre-configured loop cutoff and spending profiles (Careful, Standard, Passive). |
| ![Telemetry Export](assets/images/tokenguard-control-center.png) |  Manage proxy daemons, execute guarded scripts via tokenguard run, and toggle emergency traffic cutoff directly from the terminal. |

---

## Screenshot guide & assets checklist

All screenshots are stored in the [`assets/images/`](assets/images/) directory. If you are updating, taking new screenshots, or contributing visuals, follow this checklist:

| Asset file | Target view to capture | Recommended setup and data state |
| :--- | :--- | :--- |
| [`tokenguard-loop-detection.png`](assets/images/tokenguard-loop-detection.png) | Automatic detection and termination of alternating tool loops (A→B→A→B) before calls reach upstream LLM providers, reporting avoided cost in real time. | Run `tokenguard simulate --type alternating_loop` in an empty terminal window. Ensure table rows (steps 1–4), blocked 429 status, cycle length indicator, and avoided spend are fully visible with no prompt trailing below. |
| [`tokenguard-web-dashboard.png`](assets/images/tokenguard-web-dashboard.png) | Live request inspection, kill-switch control, budget tracking and direct configuration snippets for Python, TypeScript and shell environments. | Open dashboard at `http://127.0.0.1:8080`. Trigger at least 3 requests (2 allowed, 1 tripped loop) so traffic stream shows `[TOOL RETRY LOOP: HARD STOP]`, preserved budget ($0.01+), and code snippets tab active. |
| [`tokenguard-cli-profiles.png`](assets/images/tokenguard-cli-profiles.png) | Instant provider key detection across OpenAI, Anthropic, Gemini and DeepSeek, with pre-configured loop cutoff and spending profiles (Careful, Standard, Passive). | Launch `tokenguard init` or interactive setup with at least one active API key exported (`OPENAI_API_KEY`). Ensure profile selector highlights the `Careful` tier with loop cutoff details shown. |
| [`tokenguard-control-center.png`](assets/images/tokenguard-control-center.png) | Manage proxy daemons, execute guarded scripts via tokenguard run, and toggle emergency traffic cutoff directly from the terminal. | Start proxy daemon (`tokenguard start`). Run interactive CLI menu showing `ACTIVE & GUARDING` status, local endpoints, `Careful` profile, and numbered arrow-navigable action list. |

---

## ⚡ Core features & technical deep dive

### 1. Infinite loop & tool ping-pong circuit breaker
- **Deterministic hashing**: Computes SHA-256 hashes of normalized message arrays for sub-millisecond loop matching.
- **Sliding-window matching**: Compares incoming prompts against previous requests inside a rolling time window (default: 60s).
- **Non-poisoning architecture**: When an identical prompt is blocked with `429 Too Many Requests`, the rejection is **not** appended to the rolling window. As a result, when the agent fixes its state and sends a distinct prompt, it succeeds immediately.
- **Multi-step tool cycle detection**:
  - **Length 1 cycle ($A \to A \to A$)**: Intercepts identical tool call retries.
  - **Length 2 cycle ($A \to B \to A \to B$)**: Intercepts oscillating tool loops (e.g. `read_file` $\leftrightarrow$ `bash` execution cycles).
  - **Length 3 cycle ($A \to B \to C \to A \to B \to C$)**: Intercepts recurring three-step exploratory agent cycles.
- **Dynamic nonce & volatile ID stripping**: Automatically strips dynamic fields (`call_id`, `timestamp`, `ts`, `nonce`, `_id`, `request_id`) in `normalize_tool_arguments()` before running similarity analysis. Prevents agent framework runtime IDs or timestamps from masking repeated tool calls.

### 2. Velocity & burn-rate circuit breaker
- **Sliding window burn rate**: Beyond static prompt hashing, autonomous agents can burn budget across non-identical retry loops. TokenGuard tracks spending velocity with `--max-burn-rate` (e.g. `$0.50/min`, `$2.00/5m`, or `1.50/30s`).
- **SQLite spend hydration**: Rehydrates previous 24 hours of expenditure on proxy boot to guarantee uninterrupted budget velocity enforcement across server restarts.
- **Immediate hard cutoff**: Instantly trips before upstream execution with an HTTP 429 machine-readable payload showing current spend and breached threshold.

### 3. Context ballooning guard
- **Runaway context detection**: Autonomous agents trapped in retry loops monotonically expand their context as compiler errors and debug logs accumulate.
- **Monotonic growth evaluation**: Tracks session sequence (`session_id`). If `prompt_tokens` increases by > 300% across 3 consecutive requests without an explicit task update or cache hit, it is blocked immediately as abnormal runaway context expansion.
- **Bypass markers**: Automatically allows large context growth when accompanied by a task milestone (e.g. `x-task-status`, metadata commit status) or prompt caching hits (`x-cache-hit`).

### 4. Observation mode (`--dry-run` / shadow mode)
- **Zero-disruption production auditing**: Run TokenGuard with `--dry-run` to monitor trip conditions without severing LLM agent connections or breaking production workflows.
- **Pass-through execution**: When a circuit breaker trips in observation mode, TokenGuard logs terminal warning alerts, records the `blocked_reason` in the SQLite database, and transparently passes the request upstream (`200 OK`).
- **Audit response headers**: Injects `X-TokenGuard-Dry-Run: true` and `X-TokenGuard-Status: tripped` along with the reason and saved telemetry headers into upstream HTTP responses.

### 5. Saved tokens & cost telemetry
- **Real-time savings calculation**: On every tripped circuit breaker, TokenGuard calculates the exact prompt tokens and estimated avoided completion tokens (derived from session average history or fallback 500 completion tokens) multiplied by live model rates.
- **Custom HTTP telemetry headers**:
  - `X-TokenGuard-Status: tripped`
  - `X-TokenGuard-Reason: <reason>`
  - `X-TokenGuard-Tokens-Saved: <count>`
  - `X-TokenGuard-USD-Saved: <cost>`
- **Machine-readable JSON body**: Both the root object and nested `error` dictionary include `tokens_saved` and `usd_saved`.

### 6. True zero-latency streaming (SSE proxying)
- **Pass-through asynchronous streaming**: When `stream: true` is requested, chunks are yielded downstream immediately (< 5ms latency overhead) via `httpx.AsyncClient` and `fastapi.responses.StreamingResponse`. Never buffers responses in memory.
- **Non-blocking background accounting (`StreamAccumulator`)**: Concurrently parses native provider usage frames (e.g. OpenAI `stream_options={"include_usage": true}` or Anthropic usage blocks) without introducing backpressure to the generator loop.
- **Omitted usage fallback**: If upstream omits token usage, maintains a lightweight character accumulator to estimate tokens accurately.
- **Robust connection lifecycle**: Gracefully handles client disconnects and `asyncio.CancelledError`, immediately closing upstream sockets to prevent socket leaks.

### 7. Budget limiter and avoided cost ledger
- **Dual sliding windows**: Monitors both sliding 1-hour and 24-hour spending totals against configurable caps (e.g. `--limit 5.0` for \$5/hr).
- **Avoided cost calculation**: When a request is blocked, TokenGuard calculates how much money was saved based on the model's token rate, prompt size, and estimated completion tokens.
- **Audio and desktop notifications**: Fires a non-blocking background notification (macOS AppleScript / Linux `notify-send`) and audible terminal bell when a runaway loop is intercepted.

### 8. Dynamic real-time pricing synchronizer
- **OpenRouter Live Registry**: Automatically fetches the latest pricing tables for over 400 models from `openrouter.ai/api/v1/models` without requiring an API key.
- **Local TTL caching**: Saves cached rates to `~/.tokenguard/prices_cache.json` with a 24-hour TTL.
- **Offline-first fallback**: If internet connectivity is down, it instantly falls back to the embedded [`tokenguard/prices.json`](tokenguard/prices.json) baseline.

### 9. Multi-format telemetry exporter
- **HAR 1.2 standard**: Exports valid HTTP Archive 1.2 files compatible with Chrome DevTools, Postman, and Charles Proxy, enriched with custom `_tokenCost`, `_promptTokens`, `_completionTokens`, and `_blockedReason` fields.
- **JSONL and CSV**: Fast newline-delimited JSON and standard CSV streams for downstream data pipelines, Pandas, and SIEM logging.
- **Non-blocking WAL queries**: Reads from SQLite with `PRAGMA query_only = ON;` and `PRAGMA synchronous = NORMAL;`, ensuring zero lock contention with active proxy writers.

### 10. Zero-friction Python SDK
- **Function decorator (`@tokenguard.protect`)**: Protects synchronous or asynchronous functions with automatic proxy routing.
- **Context manager (`with tokenguard.guard():`)**: Sets environment variables (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `DEEPSEEK_BASE_URL`, etc.) within the scope and safely restores them on exit or exceptions.
- **Daemon auto-start**: Automatically starts the TokenGuard background daemon if it is not currently running.

### 11. High-end developer CLI & web dashboard
- **CLI TUI**: Built with Rich, featuring clean ASCII glyphs (`›`, `·`, `●`), zero-emoji clutter, and rock-solid column alignment that never wraps on 80-column terminals.
- **Next.js web dashboard**: Dark-mode interface with live SSE updates, distribution charts, quickstart simulations, and interactive configuration controls.

---

## Installation and Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- macOS or Linux (Windows supported via WSL)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Shvirlok/TokenGuard.git
cd TokenGuard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Configure API keys
TokenGuard detects keys automatically from your environment, `.env` file, or stored config:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
```

### 3. Launch TokenGuard

```bash
# Start TokenGuard in background daemon mode
tokenguard start -d

# Open interactive Control Center TUI
tokenguard

# Open Web Dashboard in browser
open http://127.0.0.1:8080
```

### 4. Route your existing applications & terminal
You can route any terminal session to TokenGuard in a single command without modifying source code:

```bash
# Export proxy variables for OpenAI, Anthropic, Gemini, DeepSeek
eval $(tokenguard env)

# Run any agent script transparently through TokenGuard
python my_agent.py

# When finished, restore direct upstream connections
eval $(tokenguard env --unset)
```

Or execute commands directly through the subprocess wrapper:
```bash
tokenguard run -- python my_agent.py
```

---

## 💻 Complete CLI command reference

TokenGuard provides a full suite of CLI commands for server management, script wrapping, telemetry export, live pricing updates, and interactive terminal control.

```
usage: tokenguard [-h] {start,stop,status,run,env,simulate,export,prices,kill,resume,profile,clear,logs,quickstart} ...
```

---

### 1. `tokenguard start`
Start the TokenGuard proxy server and dashboard.

```bash
tokenguard start [options]
```

| Flag / Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-d`, `--daemon` | Flag | `False` | Run TokenGuard as a background daemon process. |
| `--host` | `str` | `127.0.0.1` | Network interface to bind the proxy server to. |
| `--port` | `int` | `8080` | Port for the proxy server and dashboard. |
| `--limit` | `float` | `5.0` | Sliding hourly budget limit in USD. |
| `--daily-limit` | `float` | `50.0` | Sliding daily budget limit in USD. |
| `--max-burn-rate` | `str` | `None` | Maximum spending burn rate velocity limit (e.g. `$0.50/min`, `$2.00/5m`, `1.50/30s`). |
| `--max-repeats` | `int` | `3` | Maximum identical consecutive requests before tripping (alias for `--loop-threshold`). |
| `--loop-window` | `float` | `60.0` | Sliding time window in seconds for consecutive loop matching. |
| `--profile` | `str` | `careful` | Preset profile: `careful` (\$5/hr, 2 loops), `standard` (\$15/hr, 4 loops), `passive`. |
| `--passive` | Flag | `False` | Run in passive observability mode (no loop or budget cutoffs). |
| `--dry-run` | Flag | `False` | Run in observation mode (detect and log trips, but pass requests through upstream without 429 cutoffs). |

**Examples:**
```bash
# Start in background with strict $2/hr limit and $0.50/min burn rate velocity
tokenguard start -d --limit 2.0 --max-burn-rate '$0.50/min'

# Start in observation shadow mode to audit trips without blocking requests
tokenguard start -d --dry-run

# Start in passive monitoring mode on custom port
tokenguard start -d --port 9090 --passive
```

---

### 2. `tokenguard stop`
Stop any active background TokenGuard daemon process.

```bash
tokenguard stop [--port 8080]
```

---

### 3. `tokenguard status`
Display real-time health, daemon PID, hourly spend, avoided savings, and active API key status.

```bash
tokenguard status
```

**Output Preview:**
```text
╭───────────────────────────── TokenGuard Status ──────────────────────────────╮
│                                                                              │
│          Status:  ● ACTIVE & GUARDING                                        │
│  Proxy Base URL:  http://127.0.0.1:8080/v1                                   │
│  Live Dashboard:  http://127.0.0.1:8080                                      │
│      Daemon PID:  58871                                                      │
│  Active Profile:  Careful                                                    │
│    Hourly Spend:  $0.0016 / $5.00                                            │
│     Money Saved:  +$0.1713                                                   │
│        Requests:  7 (5 ok, 2 blocked)                                        │
│     Keys Active:  OpenAI (sk-7••••c530)                                      │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

### 4. `tokenguard run`
Execute any command or script with `OPENAI_BASE_URL`, `OPENAI_API_BASE`, and `ANTHROPIC_BASE_URL` automatically injected into the subprocess environment.

```bash
tokenguard run [options] -- <command> [args...]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | `int` | `8080` | Proxy port to route traffic through. |
| `--limit` | `float` | `5.0` | Hourly budget limit for this run. |
| `--daily-limit` | `float` | `50.0` | Daily budget limit for this run. |
| `--max-burn-rate` | `str` | `None` | Burn rate velocity limit for this run (e.g. `$0.50/min`). |
| `--max-repeats` | `int` | `3` | Maximum loop repeats before blocking. |
| `--passive` | Flag | `False` | Run in non-blocking observability mode. |
| `--dry-run` | Flag | `False` | Run in observation mode (detect and log trips, but pass requests through upstream). |

**Examples:**
```bash
# Protect a Python agent script with burn rate and hourly limit
tokenguard run --max-burn-rate '$0.50/min' --limit 5.0 -- python agent.py

# Run in observation mode to test circuit breaker thresholds against a live run
tokenguard run --dry-run -- python agent.py

# Run an npm/Node.js autonomous agent
tokenguard run -- npm start

# Protect a pytest suite executing real LLM integration tests
tokenguard run --max-repeats 2 -- pytest tests/
```

---

### 5. `tokenguard env` (Terminal Environment Injector)
Print shell environment export or unset commands to route LLM SDKs through TokenGuard.

```bash
tokenguard env [options]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--unset` | Flag | `False` | Print shell commands to unset TokenGuard environment variables. |
| `--host` | `str` | `localhost` | Proxy host address. |
| `--port` | `int` | `8080` | Proxy port. |

**Examples:**
```bash
# Route current terminal session through TokenGuard:
eval $(tokenguard env)

# Check active proxy URL in environment:
echo $OPENAI_BASE_URL
# Output: http://localhost:8080/v1

# Restore direct upstream connections:
eval $(tokenguard env --unset)

# Specify custom host/port:
eval $(tokenguard env --host 192.168.1.10 --port 9090)
```

---

### 6. `tokenguard simulate` (Loop & Tool Cycle Simulator)
Run instant circuit breaker simulation to verify loop or budget protection locally or through the proxy.

```bash
tokenguard simulate [options]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--type` | `choice` | `loop` | Simulation type: `loop`, `tool_loop`, `alternating_loop`, `budget`, `success`. |
| `--model` | `str` | `gpt-4o` | Model name for simulation. |
| `--host` | `str` | `127.0.0.1` | Proxy host address. |
| `--port` | `int` | `8080` | Proxy port. |

**Examples:**
```bash
# Simulate alternating tool ping-pong cycle (read_file <-> bash):
tokenguard simulate --type alternating_loop

# Simulate single repeated tool retry loop:
tokenguard simulate --type tool_loop

# Simulate prompt repeat loop:
tokenguard simulate --type loop
```

---

### 7. `tokenguard export`
Export recorded request telemetry directly from SQLite in WAL mode without blocking incoming proxy writes.

```bash
tokenguard export [options]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--format` | `choice` | `jsonl` | Output format: `jsonl`, `har`, `json`, `csv`. |
| `--output` | `path` | Auto-generated | Destination file path (defaults to timestamped file). |
| `--status` | `str` | `all` | Filter by status: `all`, `success`, `blocked`. |

**Examples:**
```bash
# Export to HAR 1.2 for inspection in Chrome DevTools / Postman
tokenguard export --format har --output audit_session.har

# Export blocked loop events to JSONL
tokenguard export --format jsonl --status blocked --output blocked_loops.jsonl

# Export all metrics to CSV for Excel / Pandas analysis
tokenguard export --format csv --output telemetry.csv
```

---

### 8. `tokenguard prices`
Manage model pricing registries, sync live rates from OpenRouter, and view active token costs.

```bash
tokenguard prices [options]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--update` | Flag | `False` | Force fetch live model pricing registry from OpenRouter public API. |
| `--offline` | Flag | `False` | Display local offline baseline prices from `prices.json`. |
| `--export` | `path` | `None` | Export active pricing table to a JSON file. |

**Examples:**
```bash
# Sync live rates for 400+ models from OpenRouter
tokenguard prices --update

# View offline baseline rates
tokenguard prices --offline
```

---

### 9. Reactive CLI control (`kill`, `resume`, `profile`, `clear`, `logs`)
Quickly inspect or control running TokenGuard daemons directly from the terminal without opening a browser:

```bash
# Engage emergency kill switch immediately (blocks all LLM traffic with HTTP 429)
tokenguard kill

# Disengage kill switch and resume normal traffic
tokenguard resume

# Switch active guard profile on the fly
tokenguard profile careful   # $5/hr limit, threshold 2
tokenguard profile standard  # $15/hr limit, threshold 4
tokenguard profile passive   # Observability only (no cutoffs)

# Wipe request logs and reset in-memory counters
tokenguard clear

# Stream live LLM request logs in terminal
tokenguard logs -f
```

---

### 10. `tokenguard quickstart`
Interactive setup wizard to detect API keys, configure guard profiles, and launch the proxy.

```bash
tokenguard quickstart
```

---

### Interactive CLI navigation

Running `tokenguard` without arguments opens the full interactive terminal console:

```bash
tokenguard
```

- **Navigation**: Use <kbd>↑</kbd> and <kbd>↓</kbd> arrow keys to navigate menu items.
- **Selection**: Press <kbd>Enter</kbd> to execute the highlighted action.
- **Direct access**: Type numbers <kbd>1</kbd> through <kbd>10</kbd> or press <kbd>q</kbd> to quit.

---

## Python SDK guide and code recipes

TokenGuard includes a zero-friction Python SDK allowing direct in-code protection of LLM calls without modifying base URLs manually.

### Function decorator (`@protect`)

Use `@tokenguard.protect` on synchronous or asynchronous functions:

```python
import tokenguard
from openai import OpenAI

# Protect sync function
@tokenguard.protect(max_repeats=3, budget_limit=10.0)
def generate_summary(text: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.choices[0].message.content

# Protect async function
@tokenguard.protect(max_repeats=2)
async def ask_agent_async(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

---

### Context manager (`with guard():`)

Scope protection to specific code blocks:

```python
import tokenguard
from openai import OpenAI

client = OpenAI()

# Standard synchronous block
with tokenguard.guard(max_repeats=3, budget_limit=5.0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Execute agent step"}],
    )
    print(response.choices[0].message.content)
```

---

### Framework integrations

#### 1. Native Anthropic messages API
TokenGuard automatically intercepts both Anthropic native format and OpenAI Claude calls:

```python
import tokenguard
import anthropic

client = anthropic.Anthropic()

with tokenguard.guard(max_repeats=3):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Analyze system logs"}],
    )
    print(message.content[0].text)
```

#### 2. LangChain & LangGraph
```python
import tokenguard
from langchain_openai import ChatOpenAI

with tokenguard.guard(max_repeats=3):
    llm = ChatOpenAI(model="gpt-4o")
    result = llm.invoke("Draft test suite for authentication module")
    print(result.content)
```

#### 3. DeepSeek direct integration
```python
import tokenguard
from openai import OpenAI

# DeepSeek routes automatically to https://api.deepseek.com
with tokenguard.guard(max_repeats=2):
    client = OpenAI()
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Solve combinatorial puzzle"}],
    )
    print(response.choices[0].message.content)
```

---

## REST API reference

TokenGuard exposes standard proxy routes alongside management and streaming endpoints on port `8080`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Universal OpenAI-compatible chat proxy with circuit breaker and auto-routing. |
| `POST` | `/v1/messages` | Native Anthropic Messages API proxy with loop and budget protection. |
| `GET` | `/v1/models` | List available models across all registered providers. |
| `POST` | `/v1/embeddings` | Embeddings proxy with token cost calculation and loop checks. |
| `GET` | `/health` | Server health check and active kill switch status. |
| `GET` | `/api/stats` | Aggregated metrics, sliding spend, avoided cost savings, and breaker health. |
| `GET` | `/api/requests` | Paginated request logs with status, tokens, latency, and costs. |
| `GET` | `/api/export` | Export telemetry stream (`?format=har|jsonl|json|csv&status=all|blocked`). |
| `GET` | `/api/history` | Hourly spend and request breakdown for chart rendering. |
| `GET` | `/api/stream` | Server-Sent Events (SSE) live event ticker. |
| `POST` | `/api/kill-switch` | Toggle or set emergency kill switch state (`{"active": true/false}`). |
| `POST` | `/api/config` | Dynamically update budget limits, loop threshold, and active profile. |
| `POST` | `/api/simulate` | Trigger simulated loop or success events for onboarding validation. |
| `POST` | `/api/clear-logs` | Wipe SQLite request logs and reset in-memory counters. |

### Circuit breaker 429 error format & custom headers

When TokenGuard trips due to a burn-rate breach, context ballooning, budget limit, tool ping-pong cycle, or prompt loop, it cuts off upstream calls and returns an explicit, machine-readable HTTP 429 response accompanied by custom telemetry headers:

#### Response Headers:
```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-TokenGuard-Status: tripped
X-TokenGuard-Reason: TokenGuard: infinite agent loop detected (3 identical consecutive requests)
X-TokenGuard-Tokens-Saved: 650
X-TokenGuard-USD-Saved: 0.0065
```

#### Response Body:
```json
{
  "error": {
    "message": "TokenGuard: infinite agent loop detected (3 identical consecutive requests)",
    "type": "loop_detected",
    "tokens_saved": 650,
    "usd_saved": 0.0065
  },
  "tokens_saved": 650,
  "usd_saved": 0.0065
}
```

#### Observation Mode (`--dry-run`) Headers:
When running in observation mode, the connection is **not** cut with 429; instead, the request is passed through upstream with `200 OK`, and shadow trip telemetry headers are attached:
```http
HTTP/1.1 200 OK
X-TokenGuard-Dry-Run: true
X-TokenGuard-Status: tripped
X-TokenGuard-Reason: TokenGuard intercepted alternating tool call loop (Length 2: read_file -> bash)
X-TokenGuard-Tokens-Saved: 850
X-TokenGuard-USD-Saved: 0.0085
```

#### Common Circuit Breaker Codes:
- `TOOL_PING_PONG_LOOP`: Intercepted alternating ($A \to B \to A \to B$), cyclical, or identical tool call loops.
- `burn_rate_exceeded`: Sliding window spending velocity limit breached.
- `context_ballooning_detected`: Prompt tokens expanded > 300% across 3 consecutive requests in session.
- `budget_exceeded`: Sliding hourly or daily expenditure limit exceeded.
- `loop_detected`: Identical prompt repeated consecutively.
- `kill_switch_active`: Emergency kill switch active.

---

## Model pricing matrix

TokenGuard maintains a built-in pricing registry (USD per 1,000,000 tokens) updated in real time via OpenRouter:

| Model Identifier | Provider | Input Rate ($/1M) | Output Rate ($/1M) | Auto-Detected Prefix |
| :--- | :--- | :---: | :---: | :--- |
| `gpt-4o` | OpenAI | \$2.50 | \$10.00 | `gpt-4o*` |
| `gpt-4o-mini` | OpenAI | \$0.15 | \$0.60 | `gpt-4o-mini*` |
| `o1` | OpenAI | \$15.00 | \$60.00 | `o1*` |
| `o3-mini` | OpenAI | \$1.10 | \$4.40 | `o3-mini*` |
| `claude-3-5-sonnet` | Anthropic | \$3.00 | \$15.00 | `claude-3-5-sonnet*` |
| `claude-3-7-sonnet` | Anthropic | \$3.00 | \$15.00 | `claude-3-7-sonnet*` |
| `claude-3-haiku` | Anthropic | \$0.25 | \$1.25 | `claude-3-haiku*` |
| `deepseek-chat` | DeepSeek | \$0.14 | \$0.28 | `deepseek-chat*` |
| `deepseek-reasoner` | DeepSeek | \$0.55 | \$2.19 | `deepseek-reasoner*` |
| `gemini-2.0-flash` | Google | \$0.10 | \$0.40 | `gemini-2.0-flash*` |
| `gemini-1.5-pro` | Google | \$1.25 | \$5.00 | `gemini-1.5-pro*` |
| `qwen-max` | Alibaba | \$2.80 | \$8.40 | `qwen-max*` |
| `qwen-2.5-coder` | Alibaba | \$0.20 | \$0.60 | `qwen-2.5-coder*` |
| `llama-3.3-70b` | Meta / Groq | \$0.59 | \$0.79 | `llama-3.3-70b*` |
| `codestral` | Mistral | \$0.30 | \$0.90 | `codestral*` |
| `default` | Fallback | \$1.00 | \$2.00 | Unrecognized models |

*Run `tokenguard prices --update` to refresh rates directly from OpenRouter.*

---

## Automated testing and CI

TokenGuard is thoroughly validated by an automated test suite of **106 tests** executed against multi-OS environments on GitHub Actions.

```bash
# Run complete test suite with verbose output (106 tests)
pytest -v

# Run tests with short summary
pytest -q

# Run specific test suites
pytest tests/test_guard.py tests/test_proxy.py tests/test_cli.py tests/test_db.py
```

### Test Coverage Highlights
- **Circuit Breaker Integrity**: Prompt loop detection, tool ping-pong cycle detection ($A \to B \to A \to B$), dynamic nonce stripping, burn-rate limits, and context ballooning guards.
- **Observation Mode**: Verified pass-through behavior, shadow header attachment, and SQLite blocked-reason audit logging.
- **Saved Telemetry**: Validated token count and USD pricing calculations injected into 429 response headers and JSON bodies.
- **Database & WAL Mode**: Non-destructive migrations (`ALTER TABLE requests ADD COLUMN session_id TEXT`), indexing (`idx_requests_session_id`), and concurrent WAL reader/writer isolation.
- **Multi-Provider Proxies**: OpenAI chat completions, native Anthropic messages, Anthropic-to-OpenAI adapters, and DeepSeek endpoints.
- **CLI & SDK**: Environment injector (`tokenguard env`), interactive TUI navigation, daemon lifecycle management, and `@protect` / `with guard():` decorators.

```bash
# Run complete test suite (106 tests passing)
pytest -v

# Run tests with short summary
pytest -q

# Run specific test suites
pytest tests/test_guard.py tests/test_proxy.py tests/test_cli.py tests/test_db.py
```

### GitHub Actions Workflow Matrix
- **OS Matrix**: `ubuntu-latest`, `macos-latest`
- **Python Matrix**: `3.10`, `3.11`, `3.12`
- **Verification**: Zero regression test passes, SQLite WAL isolation, dynamic price sync resilience, and SDK environment restoration.

---

## Contributing and license

Contributions are welcome! Please submit issues, feature proposals, and pull requests via GitHub.

This project is licensed under the **MIT License** &mdash; see the [`LICENSE`](LICENSE) file for details.

---

<p align="center">
  <b>TokenGuard</b> &bull; Built with FastAPI, Rich, and SQLite WAL for resilient AI agent engineering.
</p>
