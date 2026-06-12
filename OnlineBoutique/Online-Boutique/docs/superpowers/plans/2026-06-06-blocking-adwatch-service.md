# Blocking Ad Watch Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a blocking, staged reward video ad flow backed by a new AdWatchService, local MP4 creatives, Grafana monitoring, and QPS-tier load tests.

**Architecture:** `frontend` keeps the browser routes and session cookie handling, but proxies all watch lifecycle calls to a new `adwatchservice`. `adwatchservice` owns ad watch observability and forwards authoritative watch validation and coin mutation to `rewardservice`. `adservice` serves metadata for local MP4 creatives through existing gRPC fields.

**Tech Stack:** Go frontend, Python Flask/Gunicorn AdWatchService, existing Python RewardService, Java AdService, Prometheus metrics, Grafana dashboards, Docker Compose, Kubernetes manifests, JMeter.

---

## File Structure

- Create `src/adwatchservice/adwatchservice.py`: Flask facade for start/event/claim/abandon, metrics, health.
- Create `src/adwatchservice/test_adwatchservice.py`: unit tests using a fake reward client.
- Create `src/adwatchservice/requirements.txt`: Flask, Gunicorn, gevent, Prometheus client.
- Create `src/adwatchservice/Dockerfile`: Python service image.
- Modify `src/frontend/main.go`: read `ADWATCH_SERVICE_ADDR`, register abandon route.
- Modify `src/frontend/handlers.go`: proxy watch lifecycle to AdWatchService instead of RewardService.
- Modify `src/frontend/handlers_reward_test.go`: verify forwarding, completion proxy behavior, abandon proxy behavior.
- Modify `src/frontend/templates/ad.html`: expose blocking modal lifecycle and send abandon events.
- Modify `src/frontend/templates/cart.html`: mark checkout form for blocking ad submit interception.
- Copy MP4 files into `src/frontend/static/videos/`.
- Modify `src/adservice/src/main/java/hipstershop/AdService.java`: rotate local video URLs across existing known ad IDs.
- Modify `docker-compose.yml`: add `adwatchservice`, set frontend `ADWATCH_SERVICE_ADDR`.
- Create `kubernetes-manifests/adwatchservice.yaml`: deployment, service, HPA, scrape annotations.
- Modify `kubernetes-manifests/frontend.yaml`: set `ADWATCH_SERVICE_ADDR`.
- Modify `kubernetes-manifests/kustomization.yaml`: include `adwatchservice.yaml`.
- Modify `kubernetes-manifests/monitoring/prometheus-config.yaml`: scrape `adwatchservice`.
- Modify `kubernetes-manifests/monitoring-servicemonitor.yaml`: add AdWatchService ServiceMonitor.
- Modify `skaffold.yaml`: add `adwatchservice` artifact.
- Modify `docs/grafana/ad-video-stability.json`: add AdWatchService panels.
- Modify `docs/grafana/README.md`: document metrics and dashboard meaning.
- Create `tests/jmeter/ad-completion-reward-loadtest.jmx`: focused blocking ad stage claim flow.
- Create `scripts/run-jmeter-ad-completion-qps.sh`: stepped QPS wrapper with Prometheus snapshots.
- Modify `tests/jmeter/README.md`: document new test.

## Task 1: AdWatchService Facade

**Files:**
- Create: `src/adwatchservice/adwatchservice.py`
- Create: `src/adwatchservice/test_adwatchservice.py`
- Create: `src/adwatchservice/requirements.txt`
- Create: `src/adwatchservice/Dockerfile`

- [ ] **Step 1: Write failing tests**

Create `src/adwatchservice/test_adwatchservice.py` with tests for start, event,
stage claims, completion metrics, and abandon.

```python
import json
import unittest

import adwatchservice


class FakeRewardClient:
    def __init__(self):
        self.calls = []
        self.responses = {
            "/ads/watch/start": (200, {"watch_id": "watch-1", "stages": [{"stage": 1, "trigger_sec": 10, "coins": 5}]}),
            "/ads/watch/event": (200, {"watch_id": "watch-1", "max_position_ms": 30000}),
            "/earn": (200, {"coins_added": 20, "balance": 120, "stage": 3}),
        }

    def post_json(self, path, payload):
        self.calls.append((path, payload))
        return self.responses[path]


class AdWatchServiceTest(unittest.TestCase):
    def setUp(self):
        self.reward = FakeRewardClient()
        self.app = adwatchservice.create_app(reward_client=self.reward)
        self.client = self.app.test_client()

    def test_start_forwards_to_rewardservice_and_counts_success(self):
        res = self.client.post("/ads/watch/start", json={
            "session_id": "session-1",
            "ad_id": "ad-watch-001",
            "creative_id": "creative-watch-video-001",
            "campaign_id": "campaign-reward-video-demo",
            "duration_ms": 30000,
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.reward.calls[0][0], "/ads/watch/start")
        metrics = self.client.get("/metrics").data
        self.assertIn(b'adwatch_started_total{ad_id="ad-watch-001"', metrics)
        self.assertIn(b'result="success"', metrics)

    def test_stage_three_claim_counts_completion_and_coins(self):
        res = self.client.post("/ads/watch/claim", json={
            "session_id": "session-1",
            "ad_id": "ad-watch-001",
            "creative_id": "creative-watch-video-001",
            "campaign_id": "campaign-reward-video-demo",
            "stage": 3,
            "watch_id": "watch-1",
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.reward.calls[0][0], "/earn")
        metrics = self.client.get("/metrics").data
        self.assertIn(b'adwatch_completed_total{ad_id="ad-watch-001"', metrics)
        self.assertIn(b'adwatch_coins_awarded_total{ad_id="ad-watch-001",campaign_id="campaign-reward-video-demo",stage="3"} 20.0', metrics)

    def test_stage_one_claim_does_not_count_completion(self):
        self.reward.responses["/earn"] = (200, {"coins_added": 5, "balance": 105, "stage": 1})
        res = self.client.post("/ads/watch/claim", json={
            "session_id": "session-1",
            "ad_id": "ad-watch-001",
            "creative_id": "creative-watch-video-001",
            "campaign_id": "campaign-reward-video-demo",
            "stage": 1,
            "watch_id": "watch-1",
        })
        self.assertEqual(res.status_code, 200)
        metrics = self.client.get("/metrics").data
        self.assertNotIn(b"adwatch_completed_total", metrics)

    def test_abandon_counts_reason_without_reward_call(self):
        res = self.client.post("/ads/watch/abandon", json={
            "ad_id": "ad-watch-001",
            "creative_id": "creative-watch-video-001",
            "campaign_id": "campaign-reward-video-demo",
            "reason": "skip",
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.reward.calls, [])
        metrics = self.client.get("/metrics").data
        self.assertIn(b'adwatch_abandoned_total{ad_id="ad-watch-001",campaign_id="campaign-reward-video-demo",creative_id="creative-watch-video-001",reason="skip"} 1.0', metrics)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python3 -m pytest src/adwatchservice/test_adwatchservice.py -q
```

Expected: FAIL because `src/adwatchservice/adwatchservice.py` does not exist.

- [ ] **Step 3: Implement minimal service**

Create `src/adwatchservice/adwatchservice.py` with:

```python
import json
import os
import time
from functools import wraps
from urllib import error, request

from flask import Flask, Response, jsonify, request as flask_request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


KNOWN_AD_IDS = {
    "ad-hairdryer-001", "ad-tank-top-001", "ad-candle-holder-001",
    "ad-bamboo-glass-jar-001", "ad-watch-001", "ad-mug-001", "ad-loafers-001",
}
KNOWN_CREATIVE_IDS = {
    "creative-hairdryer-video-001", "creative-tank-top-video-001",
    "creative-candle-holder-video-001", "creative-bamboo-glass-jar-video-001",
    "creative-watch-video-001", "creative-mug-video-001", "creative-loafers-video-001",
}
KNOWN_CAMPAIGN_IDS = {"campaign-reward-video-demo"}
KNOWN_EVENTS = {"loadedmetadata", "playing", "timeupdate", "waiting", "ended", "error", "pause", "resume"}
KNOWN_REASONS = {"skip", "exit", "error", "checkout_continue", "unknown"}

REQUEST_LATENCY = Histogram("adwatch_request_duration_seconds", "AdWatchService request duration.", ["endpoint", "result"])
STARTED = Counter("adwatch_started_total", "Ad watch starts.", ["ad_id", "creative_id", "campaign_id", "result"])
EVENTS = Counter("adwatch_event_total", "Ad watch events.", ["event", "ad_id", "creative_id", "campaign_id", "result"])
CLAIMS = Counter("adwatch_claim_total", "Ad watch claims.", ["stage", "ad_id", "creative_id", "campaign_id", "result"])
COMPLETED = Counter("adwatch_completed_total", "Completed ad watches.", ["ad_id", "creative_id", "campaign_id", "result"])
ABANDONED = Counter("adwatch_abandoned_total", "Abandoned ad watches.", ["reason", "ad_id", "creative_id", "campaign_id"])
COINS = Counter("adwatch_coins_awarded_total", "Coins awarded through ad watches.", ["stage", "ad_id", "campaign_id"])
```

Add `HTTPRewardClient.post_json()`, bounded label helpers, `create_app()`, and
the five routes. Route mapping:

- `/ads/watch/start` forwards to RewardService `/ads/watch/start`.
- `/ads/watch/event` forwards to RewardService `/ads/watch/event`.
- `/ads/watch/claim` forwards to RewardService `/earn`.
- `/ads/watch/abandon` only records `ABANDONED`.
- `/metrics` returns `generate_latest()`.
- `/_healthz` returns `ok`.

Create `src/adwatchservice/requirements.txt`:

```text
flask==3.0.3
gevent==24.11.1
gunicorn==23.0.0
prometheus_client==0.21.1
pytest==8.3.4
```

Create `src/adwatchservice/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY adwatchservice.py .

EXPOSE 8080
CMD ["gunicorn", "-w", "1", "-k", "gevent", "--worker-connections", "400", "-b", "0.0.0.0:8080", "--timeout", "30", "--access-logfile", "-", "adwatchservice:create_app()"]
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python3 -m pytest src/adwatchservice/test_adwatchservice.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/adwatchservice
git commit -m "feat: add ad watch facade service"
```

## Task 2: Frontend Blocking Proxy And Modal Flow

**Files:**
- Modify: `src/frontend/main.go`
- Modify: `src/frontend/handlers.go`
- Modify: `src/frontend/handlers_reward_test.go`
- Modify: `src/frontend/templates/ad.html`
- Modify: `src/frontend/templates/cart.html`

- [ ] **Step 1: Write failing Go handler tests**

Add tests to `src/frontend/handlers_reward_test.go`:

```go
func TestWatchAdClaimUsesAdWatchService(t *testing.T) {
	var got map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ads/watch/claim" {
			t.Fatalf("path = %q, want /ads/watch/claim", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode adwatch payload: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"coins_added": 20, "balance": 120, "stage": 3})
	}))
	defer server.Close()

	fe := frontendServer{adWatchServiceAddr: server.Listener.Addr().String()}
	rec := httptest.NewRecorder()
	req := requestWithSession(http.MethodPost, "/ads/watch", `{"ad_id":"ad-watch-001","stage":3,"watch_id":"watch-123","style":"modal","show_in":"product"}`, "session-1")

	fe.watchAdHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got["session_id"] != "session-1" || got["watch_id"] != "watch-123" {
		t.Fatalf("payload = %#v", got)
	}
}

func TestWatchAdAbandonForwardsToAdWatchService(t *testing.T) {
	var got map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ads/watch/abandon" {
			t.Fatalf("path = %q, want /ads/watch/abandon", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode abandon payload: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]bool{"ok": true})
	}))
	defer server.Close()

	fe := frontendServer{adWatchServiceAddr: server.Listener.Addr().String()}
	rec := httptest.NewRecorder()
	req := requestWithSession(http.MethodPost, "/ads/watch/abandon", `{"ad_id":"ad-watch-001","creative_id":"creative-watch-video-001","campaign_id":"campaign-reward-video-demo","reason":"skip"}`, "session-1")

	fe.watchAdAbandonHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got["session_id"] != "session-1" || got["reason"] != "skip" {
		t.Fatalf("payload = %#v", got)
	}
}
```

- [ ] **Step 2: Run frontend tests to verify RED**

Run:

```bash
go test ./src/frontend -run 'TestWatchAdClaimUsesAdWatchService|TestWatchAdAbandonForwardsToAdWatchService' -count=1
```

Expected: FAIL because `adWatchServiceAddr` and `watchAdAbandonHandler` do not
exist.

- [ ] **Step 3: Implement frontend proxy changes**

Modify `frontendServer` in `src/frontend/main.go`:

```go
adWatchServiceAddr string
```

Map the env var with RewardService fallback:

```go
svc.adWatchServiceAddr = os.Getenv("ADWATCH_SERVICE_ADDR")
if svc.adWatchServiceAddr == "" {
	svc.adWatchServiceAddr = svc.rewardServiceAddr
}
```

Register:

```go
r.HandleFunc(baseUrl+"/ads/watch/abandon", svc.watchAdAbandonHandler).Methods(http.MethodPost)
```

In `src/frontend/handlers.go`, add `adwatchPostRaw()` mirroring
`rewardPostRaw()` but using `fe.adWatchServiceAddr`.

Change:

- `watchAdStartHandler` -> `fe.adwatchPostRaw(..., "/ads/watch/start", body)`
- `watchAdEventHandler` -> `fe.adwatchPostRaw(..., "/ads/watch/event", body)`
- `watchAdHandler` -> `fe.adwatchPostRaw(..., "/ads/watch/claim", body)`

Add:

```go
func (fe *frontendServer) watchAdAbandonHandler(w http.ResponseWriter, r *http.Request) {
	var payload map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	payload["session_id"] = sessionID(r)
	body, _ := json.Marshal(payload)
	result, statusCode, err := fe.adwatchPostRaw(r.Context(), "/ads/watch/abandon", body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(result)
}
```

- [ ] **Step 4: Implement blocking modal lifecycle**

In `src/frontend/templates/ad.html`:

- Add `let activeResolve = null;` in the script state.
- Expose `window.openRewardAdModal = function(button, options) { ... }`.
- Dispatch resolution when skipped, failed, or claimed.
- Send abandon with reason `skip`, `exit`, or `error`.
- Product page auto-open delay becomes 800 ms.

Use this shape:

```javascript
function resolveRewardFlow(outcome) {
    if (activeResolve) {
        activeResolve(outcome || {});
        activeResolve = null;
    }
    window.dispatchEvent(new CustomEvent('reward-ad-resolved', { detail: outcome || {} }));
}

async function sendAbandon(reason) {
    if (!activeButton) return;
    try {
        await fetch('{{$.baseUrl}}/ads/watch/abandon', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                watch_id: watchId || '',
                ad_id: activeButton.dataset.adId || 'default-ad',
                creative_id: activeButton.dataset.creativeId || 'unknown',
                campaign_id: activeButton.dataset.campaignId || 'unknown',
                reason: reason || 'unknown'
            })
        });
    } catch (err) {}
}
```

In `src/frontend/templates/cart.html`, add an id:

```html
<form id="cart-checkout-form" class="cart-checkout-form" action="{{ $.baseUrl }}/cart/checkout" method="POST">
```

Then add a script after `ad_modal` is rendered that intercepts submit, opens the
first `.reward-ad-button`, and submits after resolution:

```javascript
document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('cart-checkout-form');
    if (!form || !window.openRewardAdModal) return;
    var submittedAfterAd = false;
    form.addEventListener('submit', function (event) {
        if (submittedAfterAd) return;
        var btn = document.querySelector('.reward-ad-button');
        if (!btn) return;
        event.preventDefault();
        window.openRewardAdModal(btn, { reason: 'checkout' }).then(function () {
            submittedAfterAd = true;
            form.submit();
        });
    });
});
```

- [ ] **Step 5: Run frontend tests to verify GREEN**

Run:

```bash
go test ./src/frontend -run 'TestWatchAd|TestWatchStart|TestWatchEvent' -count=1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/main.go src/frontend/handlers.go src/frontend/handlers_reward_test.go src/frontend/templates/ad.html src/frontend/templates/cart.html
git commit -m "feat: route blocking ad flow through adwatchservice"
```

## Task 3: Local MP4 Creatives And Deployment

**Files:**
- Copy: `../*.mp4` to `src/frontend/static/videos/`
- Modify: `src/adservice/src/main/java/hipstershop/AdService.java`
- Modify: `docker-compose.yml`
- Create: `kubernetes-manifests/adwatchservice.yaml`
- Modify: `kubernetes-manifests/frontend.yaml`
- Modify: `kubernetes-manifests/kustomization.yaml`
- Modify: `kubernetes-manifests/monitoring/prometheus-config.yaml`
- Modify: `kubernetes-manifests/monitoring-servicemonitor.yaml`
- Modify: `skaffold.yaml`

- [ ] **Step 1: Copy MP4 assets**

Run:

```bash
mkdir -p src/frontend/static/videos
cp ../88327a6271fdb0281d765bdb88f4a190.mp4 src/frontend/static/videos/
cp ../d13ceed2747033502d873f1009666eab.mp4 src/frontend/static/videos/
cp ../fb2f91885a08e2539e985ec2e5333b6a.mp4 src/frontend/static/videos/
```

Expected: `find src/frontend/static/videos -name '*.mp4'` prints three files.

- [ ] **Step 2: Update AdService video URLs**

In `AdService.java`, replace the single external video URL with:

```java
private static final String VIDEO_URL_PRIMARY =
    "/static/videos/88327a6271fdb0281d765bdb88f4a190.mp4";
private static final String VIDEO_URL_SECONDARY =
    "/static/videos/d13ceed2747033502d873f1009666eab.mp4";
private static final String VIDEO_URL_TERTIARY =
    "/static/videos/fb2f91885a08e2539e985ec2e5333b6a.mp4";
```

Use the three constants across existing known ad builders while preserving
current `ad_id`, `creative_id`, `campaign_id`, and `duration_ms`.

- [ ] **Step 3: Add local Docker Compose service**

In `docker-compose.yml`, set frontend env:

```yaml
- ADWATCH_SERVICE_ADDR=adwatchservice:8080
```

Add dependency:

```yaml
- adwatchservice
```

Add service:

```yaml
adwatchservice:
  build: ./src/adwatchservice
  image: online-boutique/adwatchservice:local
  environment:
    - PORT=8080
    - REWARD_SERVICE_ADDR=rewardservice:8080
  depends_on:
    - rewardservice
  restart: unless-stopped
```

- [ ] **Step 4: Add Kubernetes service**

Create `kubernetes-manifests/adwatchservice.yaml` with deployment, service,
service account, and HPA. Use image `adwatchservice:latest`, container port
8080, `REWARD_SERVICE_ADDR=rewardservice:80`, and Prometheus annotations for
`/metrics`.

Add `- adwatchservice.yaml` to `kubernetes-manifests/kustomization.yaml`.

Add frontend env in `kubernetes-manifests/frontend.yaml`:

```yaml
- name: ADWATCH_SERVICE_ADDR
  value: "adwatchservice:80"
```

Add Skaffold artifact:

```yaml
- image: adwatchservice
  context: src/adwatchservice
```

- [ ] **Step 5: Add Prometheus scrape config**

In `kubernetes-manifests/monitoring/prometheus-config.yaml`, add a scrape job
matching pod label `app=adwatchservice`, address `$pod_ip:8080`, path
`/metrics`, and labels `namespace`, `pod`, `app`.

In `kubernetes-manifests/monitoring-servicemonitor.yaml`, add a ServiceMonitor
for selector `app: adwatchservice`, port `http`, path `/metrics`.

- [ ] **Step 6: Verify manifests and Java formatting**

Run:

```bash
./scripts/verify-monitoring-assets.sh
```

Expected: PASS.

Run:

```bash
./src/adservice/gradlew -p src/adservice test
```

Expected: PASS or no tests with successful Gradle exit.

- [ ] **Step 7: Commit**

```bash
git add src/frontend/static/videos src/adservice/src/main/java/hipstershop/AdService.java docker-compose.yml kubernetes-manifests/adwatchservice.yaml kubernetes-manifests/frontend.yaml kubernetes-manifests/kustomization.yaml kubernetes-manifests/monitoring/prometheus-config.yaml kubernetes-manifests/monitoring-servicemonitor.yaml skaffold.yaml
git commit -m "feat: deploy adwatchservice with local video creatives"
```

## Task 4: Grafana Dashboard And Docs

**Files:**
- Modify: `docs/grafana/ad-video-stability.json`
- Modify: `docs/grafana/README.md`

- [ ] **Step 1: Add dashboard panels**

Update `docs/grafana/ad-video-stability.json` with panels using these PromQL
queries:

```promql
sum(rate(adwatch_completed_total{result="success"}[5m])) / clamp_min(sum(rate(adwatch_started_total{result="success"}[5m])), 0.001) * 100
```

```promql
sum by (reason) (rate(adwatch_abandoned_total[5m])) * 60
```

```promql
sum by (stage, result) (rate(adwatch_claim_total[5m])) * 60
```

```promql
sum by (stage, ad_id) (rate(adwatch_coins_awarded_total[5m])) * 60
```

```promql
histogram_quantile(0.99, sum by (le, endpoint) (rate(adwatch_request_duration_seconds_bucket[5m])))
```

- [ ] **Step 2: Document metrics**

Add `adwatchservice` to `docs/grafana/README.md` dashboard and metrics sections.
Include the completion-rate formula and note that stage 3 success is a
completion.

- [ ] **Step 3: Verify dashboards**

Run:

```bash
./scripts/verify-monitoring-assets.sh
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/grafana/ad-video-stability.json docs/grafana/README.md
git commit -m "feat: add adwatch grafana metrics"
```

## Task 5: Blocking Ad QPS Load Test

**Files:**
- Create: `tests/jmeter/ad-completion-reward-loadtest.jmx`
- Create: `scripts/run-jmeter-ad-completion-qps.sh`
- Modify: `tests/jmeter/README.md`

- [ ] **Step 1: Create focused JMeter plan**

Create a JMeter test plan with these properties:

```text
PROTOCOL=http
HOST=127.0.0.1
PORT=8080
THREADS=50
RAMP_SECONDS=30
DURATION_SECONDS=300
TARGET_QPM=6000
AD_ID=ad-watch-001
CREATIVE_ID=creative-watch-video-001
CAMPAIGN_ID=campaign-reward-video-demo
CLAIM_STAGE=3
WATCH_WAIT_MS=31000
SESSION_PREFIX=jmeter-ad
CHECKOUT_AFTER_AD=false
```

The sampler sequence:

1. `POST /ads/watch/start` with ad metadata and `duration_ms=30000`.
2. Extract `watch_id`.
3. Wait `${WATCH_WAIT_MS}`.
4. `POST /ads/watch/event` with `event=timeupdate` and
   `position_ms=${CLAIM_STAGE * 10000}`.
5. If `${CLAIM_STAGE}` is `3`, also send `event=ended` and `position_ms=30000`.
6. `POST /ads/watch` with `stage=${CLAIM_STAGE}` and `watch_id`.
7. If `CHECKOUT_AFTER_AD=true`, run add-to-cart and checkout samplers.

- [ ] **Step 2: Create wrapper script**

Create `scripts/run-jmeter-ad-completion-qps.sh` based on the existing reward
closed-loop wrapper. It must accept:

```text
--steps <csv>
--claim-stage <1|2|3>
--watch-wait-ms <n>
--checkout-after-ad
--prometheus-url <url>
--prometheus-port-forward
--grafana-url <url>
--grafana-port-forward
--capture-screenshots
--dry-run
```

For each step, write a JTL, HTML report, `summary.md`, `summary.csv`, and these
Prometheus snapshots:

```promql
sum(rate(adwatch_started_total{result="success"}[5m]))
sum(rate(adwatch_completed_total{result="success"}[5m])) / clamp_min(sum(rate(adwatch_started_total{result="success"}[5m])), 0.001) * 100
sum by (stage, result) (rate(adwatch_claim_total[5m]))
sum by (stage, ad_id) (rate(adwatch_coins_awarded_total[5m]))
histogram_quantile(0.99, sum by (le, endpoint) (rate(adwatch_request_duration_seconds_bucket[5m])))
sum by (endpoint) (rate(ratelimit_rejected_total{endpoint=~"watch_start|watch_event|earn"}[5m]))
```

- [ ] **Step 3: Document load test**

In `tests/jmeter/README.md`, add an "Ad Completion QPS Plan" section with
example:

```bash
scripts/run-jmeter-ad-completion-qps.sh \
  --port-forward \
  --prometheus-port-forward \
  --grafana-port-forward \
  --capture-screenshots \
  --threads 300 \
  --duration-seconds 300 \
  --steps 100,1000,2000 \
  --claim-stage 3
```

Explain that `--steps` is HTTP sample QPS, and high QPS should run in cluster or
with distributed JMeter.

- [ ] **Step 4: Dry-run verification**

Run:

```bash
scripts/run-jmeter-ad-completion-qps.sh --dry-run --steps 100,1000 --claim-stage 3 --threads 300 --duration-seconds 60
```

Expected: prints the JMeter commands for both steps and exits 0.

- [ ] **Step 5: Commit**

```bash
git add tests/jmeter/ad-completion-reward-loadtest.jmx scripts/run-jmeter-ad-completion-qps.sh tests/jmeter/README.md
git commit -m "feat: add blocking ad completion load test"
```

## Final Verification

Run:

```bash
python3 -m pytest src/adwatchservice/test_adwatchservice.py -q
go test ./src/frontend -run 'TestWatchAd|TestWatchStart|TestWatchEvent' -count=1
python3 -m pytest src/rewardservice/test_rewardservice.py -q
./scripts/verify-monitoring-assets.sh
scripts/run-jmeter-ad-completion-qps.sh --dry-run --steps 100,1000 --claim-stage 3 --threads 300 --duration-seconds 60
```

Expected: all commands exit 0.
