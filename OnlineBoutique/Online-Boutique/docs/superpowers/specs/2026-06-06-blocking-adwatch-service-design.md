# Blocking Ad Watch Service Design

## Context

The current `dev` branch already contains `qsl/feature/jsy`. The branch tip
`qsl/feature/jsy` is an ancestor of `HEAD`, so no merge is needed before this
work.

The app already has:

- `adservice`: returns ad metadata through gRPC.
- `frontend`: renders ad placements and proxies ad watch requests.
- `rewardservice`: owns coin balance, watch session validation, staged rewards,
  and Prometheus metrics.
- JMeter and Grafana assets for reward closed-loop testing.

The missing behavior is a blocking ad watch flow backed by local MP4 creatives,
an explicit ad watch service boundary, clear completion metrics, and load-test
reporting for QPS tiers such as 100, 1000, and higher.

## Requirements

- Register the three local MP4 files as reward video creatives.
- Product detail pages automatically open the blocking ad modal.
- Cart checkout must trigger a blocking ad before form submission.
- The user can skip or close the ad flow.
- Stage 1, stage 2, and stage 3 can each be claimed, with different coin
  amounts.
- A user receives coins only for a stage that has been fully reached and
  claimed.
- Skipping or exiting before a stage is reached produces no coins for that
  stage.
- Stage 3 represents completion for completion-rate monitoring.
- Ad watch handling should be exposed through a separate service boundary.
- Grafana should show starts, skips, exits, stage claims, completions,
  completion rate, coin issuance, latency, and error/limit impact.
- Load tests should exercise a full blocking ad flow and summarize business
  impact at QPS tiers.

## Approach Options

### Option 1: Only Modify RewardService

Keep all watch and metrics endpoints in `rewardservice`, point the frontend
directly to it, and add missing metrics there.

Trade-off: smallest code change, but it does not satisfy the separate ad service
boundary.

### Option 2: Add AdWatchService Facade

Add a new `adwatchservice` HTTP service. It owns the browser-facing ad watch API,
skip/exit metrics, completion metrics, and forwarding to `rewardservice` for
authoritative watch session validation and coin issuance.

Trade-off: satisfies the service boundary with modest risk. Existing
RewardService Lua and Redis atomicity remain unchanged.

### Option 3: Fully Move Watch State To AdWatchService

Move watch sessions and progress verification from `rewardservice` into
`adwatchservice`, then call a new reward ledger endpoint for coin increments.

Trade-off: cleanest long-term ownership, but high-risk because it requires
rewriting the proven watch validation and coin ledger contracts.

## Selected Design

Use Option 2.

`adwatchservice` becomes the single frontend target for ad watch operations:

- `POST /ads/watch/start`
- `POST /ads/watch/event`
- `POST /ads/watch/claim`
- `POST /ads/watch/abandon`
- `GET /metrics`
- `GET /healthz`

The service forwards start, progress event, and claim calls to `rewardservice`.
It records ad-specific observability and maps claim stage 3 success into a
completion counter. It records skip and exit events independently because those
do not need coin ledger mutation.

`rewardservice` remains the source of truth for:

- Valid ad metadata tuples.
- Watch progress plausibility.
- Stage eligibility.
- Cooldown and duplicate prevention.
- Coin balance mutation.

## Local MP4 Creatives

Copy the three MP4 files from the repository parent directory into:

- `src/frontend/static/videos/88327a6271fdb0281d765bdb88f4a190.mp4`
- `src/frontend/static/videos/d13ceed2747033502d873f1009666eab.mp4`
- `src/frontend/static/videos/fb2f91885a08e2539e985ec2e5333b6a.mp4`

Update `adservice` to rotate these URLs across known ad IDs by setting each
ad's `video_url` to `/static/videos/<file>.mp4`. Keep the existing ad IDs and
creative IDs where possible so RewardService validation remains stable.

## Frontend Behavior

The modal remains the blocking surface. While it is visible, the user cannot
interact with the underlying page. The skip button exits the ad flow, sends an
abandon event, and awards no new coins.

Product detail:

- Automatically open the blocking modal after page entry.
- Use the first available reward ad button.
- Do not block page render; block interaction once the modal opens.

Cart checkout:

- Intercept checkout form submit.
- Open the blocking modal before submit.
- After the user claims a stage or skips, allow the original form submit to
  continue.
- If the user closes the modal before claiming, the checkout can continue but no
  ad coins are awarded.

The staged reward behavior stays:

- Stage 1 at 10 seconds: 5 coins.
- Stage 2 at 20 seconds: 12 coins.
- Stage 3 at 30 seconds: 20 coins and completion.

## Metrics

`adwatchservice` exposes:

- `adwatch_request_duration_seconds{endpoint,result}`
- `adwatch_started_total{ad_id,creative_id,campaign_id,result}`
- `adwatch_event_total{event,ad_id,creative_id,campaign_id,result}`
- `adwatch_claim_total{stage,ad_id,creative_id,campaign_id,result}`
- `adwatch_completed_total{ad_id,creative_id,campaign_id,result}`
- `adwatch_abandoned_total{reason,ad_id,creative_id,campaign_id}`
- `adwatch_coins_awarded_total{stage,ad_id,campaign_id}`

Completion rate PromQL:

```promql
sum(rate(adwatch_completed_total{result="success"}[5m]))
/
clamp_min(sum(rate(adwatch_started_total{result="success"}[5m])), 0.001)
* 100
```

Coin impact PromQL:

```promql
sum by (stage, ad_id) (rate(adwatch_coins_awarded_total[5m])) * 60
```

## Grafana

Update `docs/grafana/ad-video-stability.json` with panels for:

- 5m completion rate.
- Watch starts per second.
- Abandon reason breakdown.
- Stage claim result rate.
- Stage coin issuance rate.
- RewardService p95/p99 for watch endpoints.
- AdWatchService p95/p99 and request rate.

Update `docs/grafana/README.md` to document the new service and metrics.

## Load Testing

Add a focused JMeter plan for the blocking ad flow:

1. Start watch.
2. Send playback events up to a configurable stage.
3. Claim stage 1, 2, or 3.
4. Optionally submit cart checkout after the blocking flow.

Add a wrapper script that supports:

- `--steps 100,1000,2000`
- `--claim-stage 1|2|3`
- `--abandon-ratio`
- Prometheus snapshots for completion rate, stage claim success, coins/min,
  latency, 429, and 5xx.

High QPS caveat: local port-forward runs are useful for smoke and medium tiers.
For 1000+ QPS, use an in-cluster or distributed runner because local
port-forward can become the bottleneck.

## Testing

Use TDD for code changes.

Unit tests:

- `adwatchservice` forwards session, metadata, and stage to `rewardservice`.
- Claim stage 3 success increments completion metrics.
- Stage 1 and stage 2 claims do not increment completion metrics.
- Skip/exit increments abandon metrics and does not call reward claim.
- Frontend checkout submit is intercepted until the blocking ad modal resolves.
- Product page auto-open still starts a watch session.

Verification:

- Run `go test ./src/frontend`.
- Run `python -m pytest src/rewardservice`.
- Run `python -m pytest src/adwatchservice`.
- Run the JMeter wrapper in dry-run mode.
- Run monitoring asset verification.

## Out Of Scope

- Moving the coin ledger out of `rewardservice`.
- Making video playback mandatory before checkout can complete.
- Adding DRM or browser anti-tamper beyond the existing server-side watch
  progress plausibility checks.
