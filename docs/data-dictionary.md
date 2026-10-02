# Synthetic data dictionary (all amounts BDT ৳, no PII)

Seed default 42. Chronological split: last 20% by timestamp → `test`, never trained on.

## customers.csv
`customer_id` (C######), `age_group` (18-25/26-35/36-50/50+), `district`
(Dhaka 35%, Chattogram 15%, Sylhet/Khulna/Rajshahi 10%, Barishal/Rangpur 7%, Mymensingh 6%),
`account_age_days` (30–1500), `avg_balance` (lognormal, median ~5k).

## devices.csv
`device_id` (D###### registry, DX###### fraud-new, DXL###### legit-new),
`customer_id`, `first_seen`. Most customers own 1 device, some 2.

## transactions.csv
`txn_id` (T#######, time-ordered), `sender`, `receiver`, `amount`,
`type` (P2P/merchant/bill/cash-out/salary-in), `timestamp` (2026-08, ISO-8601),
`device_id`, `location` (district), `channel` (app 70%/ussd 20%/agent 10%),
`is_fraud` (~4%), `fraud_type` (none/scam/ato/mule/agent),
`password_reset_flag`, `split` (train/test).

## Injected patterns + hardening noise
- Normal: daytime mostly, same device/location, ~1.8k median. Noise: 5% legit
  new-device, 8% night, 3% round amounts, 1% resets, 5% large cash-outs,
  2% travel, plus popular-merchant legit fan-in bursts (15–30 payments/hour).
- Scam 40%: urgent P2P to new receiver, usually night + new device. Overlap:
  30% small 3–12k amounts, 25% daytime, 20% known device.
- ATO 25%: location jump + device change + velocity burst, usually reset flag.
  Overlap: 20% no flag, 15% normal amounts, 10% short 2-txn bursts.
- Mule 20%: 10-wallet fan-in → 1 collector within 1h, usually round amounts.
  Overlap: 25% near-round off-by amounts.
- Agent 15%: large cash-out via agent. Overlap: 30% smaller 25–45k.

## Features (`features/build.py`, 26 cols)
Velocity 1h/24h counts+sums, time-since-last, amount vs user avg (capped 20x),
new receiver/device/location flags, location jump vs home district, round-amount,
reset flag, receiver fan-in counts, channel/type flags, account age, balance.
All past-only (no leakage); same logic in batch builder and `HistoryStore.featurize`.
