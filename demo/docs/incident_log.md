# Incident Log — FY2024

All P0 and P1 incidents are logged here. P0 = customer-impacting outage.
P1 = degraded service. MTTR = mean time to resolution.

## Americas

Infrastructure incidents affecting the Americas region (US-EAST-1, US-WEST-2, SA-EAST-1).

### P0 Incidents

| ID       | Date       | Service            | Root Cause                          | MTTR (min) | Impact                        |
|----------|------------|--------------------|-------------------------------------|------------|-------------------------------|
| AM-P0-001 | 2024-01-08 | Auth service       | Certificate expiry not auto-renewed | 94         | Login failures, 12k users     |
| AM-P0-002 | 2024-01-22 | Database primary   | Disk full on write replica          | 47         | Write failures, 3.2k users    |
| AM-P0-003 | 2024-02-14 | API gateway        | Config push broke rate-limit rules  | 31         | 429 errors, 8.1k users        |
| AM-P0-004 | 2024-02-29 | Search service     | Elasticsearch OOM on index rebuild  | 112        | Search down, 19k users        |
| AM-P0-005 | 2024-03-11 | Object storage     | S3 bucket policy misconfiguration   | 22         | File upload failures, 4k users|
| AM-P0-006 | 2024-03-28 | CDN edge           | Origin pull loop after config change| 68         | Static assets 404, 31k users  |
| AM-P0-007 | 2024-04-09 | Notification queue | RabbitMQ partition after upgrade    | 155        | Email alerts dropped, 7k users|
| AM-P0-008 | 2024-04-23 | Auth service       | Redis eviction under memory pressure| 83         | Session drops, 9.4k users     |
| AM-P0-009 | 2024-05-07 | Data pipeline      | Kafka consumer lag spike            | 44         | Stale dashboards, 2.8k users  |
| AM-P0-010 | 2024-05-19 | API gateway        | TLS handshake timeout after patch   | 29         | API errors, 6.3k users        |
| AM-P0-011 | 2024-06-03 | Database replica   | Replication lag cascade             | 61         | Read failures, 5.1k users     |
| AM-P0-012 | 2024-06-17 | Scheduler          | Cron worker deadlock                | 38         | Batch jobs stopped, 1.2k users|
| AM-P0-013 | 2024-07-02 | Search service     | Index corruption after failover     | 142        | Search down, 22k users        |
| AM-P0-014 | 2024-07-15 | CDN edge           | DNS misconfiguration at provider    | 76         | Latency spike, 44k users      |
| AM-P0-015 | 2024-07-29 | Auth service       | SAML assertion clock skew           | 19         | SSO failures, 3.8k users      |
| AM-P0-016 | 2024-08-12 | Object storage     | Cross-region replication disabled   | 51         | Data access errors, 2.1k      |
| AM-P0-017 | 2024-08-26 | Notification queue | Dead-letter queue overflow          | 33         | Notifications delayed, 11k    |
| AM-P0-018 | 2024-09-10 | Data pipeline      | Schema migration broke ingest       | 89         | Missing data, 4.4k users      |
| AM-P0-019 | 2024-09-24 | API gateway        | Upstream timeout cascade            | 42         | Slow API, 18k users           |
| AM-P0-020 | 2024-10-08 | Database primary   | Autovacuum lock contention          | 57         | Write slowdown, 7.9k users    |
| AM-P0-021 | 2024-10-22 | Scheduler          | Executor pool exhausted             | 24         | Job queue frozen, 890 users   |
| AM-P0-022 | 2024-11-05 | Auth service       | Dependency service timeout          | 67         | Auth slow, 14k users          |
| AM-P0-023 | 2024-11-19 | Search service     | Shard allocation failure            | 35         | Partial search, 8.2k users    |
| AM-P0-024 | 2024-12-03 | CDN edge           | Cert renewal race condition         | 48         | HTTPS errors, 26k users       |
| AM-P0-025 | 2024-12-17 | Data pipeline      | Connector memory leak               | 73         | Stale data, 3.3k users        |

## EMEA

Infrastructure incidents affecting the EMEA region (EU-WEST-1, EU-CENTRAL-1, ME-SOUTH-1).

### P0 Incidents

| ID       | Date       | Service            | Root Cause                          | MTTR (min) | Impact                        |
|----------|------------|--------------------|-------------------------------------|------------|-------------------------------|
| EM-P0-001 | 2024-01-11 | Auth service       | OAuth provider outage               | 124        | Login failures, 8.2k users    |
| EM-P0-002 | 2024-01-30 | Database primary   | Network partition between AZs       | 88         | Write failures, 1.9k users    |
| EM-P0-003 | 2024-02-18 | API gateway        | Misconfigured WAF rule blocked GETs | 41         | API errors, 5.4k users        |
| EM-P0-004 | 2024-03-04 | Object storage     | GDPR retention policy broke deletes | 63         | Delete errors, 920 users      |
| EM-P0-005 | 2024-03-19 | Search service     | Mapping explosion on new index      | 97         | Search errors, 11k users      |
| EM-P0-006 | 2024-04-02 | CDN edge           | BGP route leak at EU PoP            | 211        | Latency 10x, 38k users        |
| EM-P0-007 | 2024-04-16 | Notification queue | SMTP relay quota exceeded           | 37         | Email failures, 4.8k users    |
| EM-P0-008 | 2024-05-01 | Data pipeline      | Time-zone handling broke EU dates   | 52         | Stale dashboards, 2.1k users  |
| EM-P0-009 | 2024-05-14 | Auth service       | Token signing key rotation failed   | 44         | Auth errors, 6.7k users       |
| EM-P0-010 | 2024-05-28 | Database replica   | Storage backend I/O saturation      | 79         | Slow reads, 3.4k users        |
| EM-P0-011 | 2024-06-11 | API gateway        | Load balancer health check flap     | 28         | Intermittent errors, 9.1k     |
| EM-P0-012 | 2024-06-25 | Scheduler          | NTP drift caused job overlap        | 33         | Duplicate jobs, 440 users     |
| EM-P0-013 | 2024-07-09 | Search service     | Replica promotion during snapshot   | 108        | Search down, 14k users        |
| EM-P0-014 | 2024-07-23 | CDN edge           | SSL cert chain incomplete           | 56         | HTTPS errors, 19k users       |
| EM-P0-015 | 2024-08-06 | Object storage     | Multipart upload abort loop         | 31         | Upload failures, 1.4k users   |
| EM-P0-016 | 2024-08-20 | Data pipeline      | Connector version mismatch          | 74         | Missing data, 2.8k users      |
| EM-P0-017 | 2024-09-03 | Auth service       | LDAP directory sync timeout         | 49         | Auth slow, 7.3k users         |
| EM-P0-018 | 2024-09-17 | Database primary   | Long-running query held locks       | 62         | Write timeouts, 4.2k users    |
| EM-P0-019 | 2024-10-01 | Notification queue | Queue consumer crash loop           | 38         | Alerts delayed, 5.6k users    |
| EM-P0-020 | 2024-10-15 | CDN edge           | Origin failover misconfigured       | 83         | Errors on failover, 22k users |

## APAC

Infrastructure incidents affecting the APAC region (AP-SOUTHEAST-1, AP-NORTHEAST-1, AP-SOUTH-1).

### P0 Incidents

| ID       | Date       | Service            | Root Cause                          | MTTR (min) | Impact                        |
|----------|------------|--------------------|-------------------------------------|------------|-------------------------------|
| AP-P0-001 | 2024-01-15 | Auth service       | Regional provider maintenance gap  | 178        | Login failures, 3.1k users    |
| AP-P0-002 | 2024-02-01 | Database primary   | Disk IOPS throttling on burstable   | 66         | Write slowdown, 890 users     |
| AP-P0-003 | 2024-02-22 | API gateway        | Upstream circuit breaker open       | 34         | API errors, 2.2k users        |
| AP-P0-004 | 2024-03-08 | CDN edge           | PoP capacity exceeded in SG         | 121        | Latency spike, 14k users      |
| AP-P0-005 | 2024-03-25 | Search service     | JVM GC pause during peak traffic    | 58         | Search timeout, 4.8k users    |
| AP-P0-006 | 2024-04-11 | Object storage     | Cross-border data transfer blocked  | 144        | Upload failures, 1.1k users   |
| AP-P0-007 | 2024-04-28 | Data pipeline      | Regional endpoint unavailable       | 47         | Stale dashboards, 1.6k users  |
| AP-P0-008 | 2024-05-12 | Auth service       | MFA provider rate limited           | 39         | MFA failures, 2.9k users      |
| AP-P0-009 | 2024-05-27 | Notification queue | SMS gateway outage in JP            | 91         | SMS failures, 3.4k users      |
| AP-P0-010 | 2024-06-10 | Database replica   | Network jitter caused split-brain   | 113        | Read errors, 1.8k users       |
| AP-P0-011 | 2024-06-24 | API gateway        | TLS version mismatch with proxy     | 26         | API errors, 980 users         |
| AP-P0-012 | 2024-07-08 | CDN edge           | Route propagation delay AP PoPs     | 87         | Latency 5x, 19k users         |
| AP-P0-013 | 2024-07-22 | Search service     | Insufficient replicas for traffic   | 74         | Search slow, 6.1k users       |
| AP-P0-014 | 2024-08-05 | Scheduler          | Time-zone off by 1hr post-DST       | 42         | Job timing off, 330 users     |
| AP-P0-015 | 2024-08-19 | Data pipeline      | APAC ingest endpoint down           | 108        | Data gap, 2.2k users          |
