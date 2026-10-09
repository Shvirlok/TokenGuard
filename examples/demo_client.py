"""TokenGuard Demo Client & Simulation Script.

Demonstrates drop-in replacement with standard OpenAI Python client or HTTP calls.
"""

from __future__ import annotations

import time
import httpx

PROXY_URL = "http://127.0.0.1:8080"


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"🔹 {title}")
    print("=" * 60)


def run_demo():
    print("\n🛡️  TokenGuard Demo Client Initializing...")
    client = httpx.Client(base_url=PROXY_URL, timeout=30.0)

    # 1. Health Check
    print_section("1. Proxy Health Check")
    try:
        res = client.get("/health")
        print(f"Response ({res.status_code}): {res.json()}")
    except Exception as e:
        print(f"❌ Could not connect to TokenGuard on {PROXY_URL}. Ensure 'tokenguard start' is running.")
        return

    # 2. Check Live Stats
    print_section("2. Current Dashboard Stats")
    stats = client.get("/api/stats").json()
    print(f"Hourly Spend: ${stats.get('total_spent', 0):.4f} / ${stats.get('hourly_limit', 5):.2f}")
    print(f"Total Requests Logged: {stats.get('total_requests', 0)}")
    print(f"Estimated Savings: ${stats.get('saved_cost_estimate', 0):.4f}")

    # 3. Simulate Normal Chat Requests
    print_section("3. Sending Normal Requests")
    for i in range(1, 3):
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Task number {i}: summarize python async patterns"}],
        }
        res = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer sk-demo-token"})
        print(f"Request #{i} Status: {res.status_code}")
        if res.status_code == 429:
            print(f"   Circuit Breaker: {res.json()}")
        time.sleep(0.3)

    # 4. Simulate Infinite Agent Loop (3 consecutive identical requests)
    print_section("4. Simulating Infinite Agent Loop (3 Identical Consecutive Requests)")
    loop_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Rogue agent repeating identical prompt sequence"}],
    }

    for attempt in range(1, 4):
        res = client.post("/v1/chat/completions", json=loop_payload, headers={"Authorization": "Bearer sk-demo-token"})
        print(f"Attempt #{attempt} Status: {res.status_code}")
        if res.status_code == 429:
            print(f"   🛑 LOOP DETECTED & TRIPPED: {res.json()}")
        else:
            print(f"   Passed through (attempt {attempt}/3)")
        time.sleep(0.2)

    # 4b. Verify Immediate Recovery for New Distinct Prompt
    print_section("4b. Verifying Immediate Recovery on Distinct Prompt")
    new_prompt_payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Completely new prompt after loop was blocked"}],
    }
    recovery_res = client.post("/v1/chat/completions", json=new_prompt_payload, headers={"Authorization": "Bearer sk-demo-token"})
    print(f"New Prompt Status: {recovery_res.status_code} (Expected: 401 upstream or 200, NOT 429 loop_detected)")
    if recovery_res.status_code != 429:
        print("   ✅ Distinct prompt passed through without loop blocking!")

    # 5. Test Kill Switch Toggle
    print_section("5. Testing Manual Kill Switch")
    # Turn ON
    client.post("/api/kill-switch", json={"active": True})
    print("Kill switch activated!")

    blocked_res = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Should be blocked by kill switch"}]},
        headers={"Authorization": "Bearer sk-demo-token"}
    )
    print(f"Kill Switch Request Status: {blocked_res.status_code}")
    print(f"Response: {blocked_res.json()}")

    # Turn OFF
    client.post("/api/kill-switch", json={"active": False})
    print("Kill switch deactivated!")

    # 6. Final Stats
    print_section("6. Final Dashboard Stats")
    final_stats = client.get("/api/stats").json()
    print(f"Total Requests: {final_stats.get('total_requests', 0)}")
    print(f"Blocked Requests: {final_stats.get('blocked_requests', 0)}")
    print(f"Blocked Loops: {final_stats.get('blocked_loop_count', 0)}")
    print(f"Money Saved: ${final_stats.get('saved_cost_estimate', 0):.4f}")
    print("\n✅ Demo simulation complete! Open http://127.0.0.1:8080 to view the live dashboard.")


if __name__ == "__main__":
    run_demo()
