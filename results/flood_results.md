Flood test: 40 simulated users, spawn rate 10/s, 60s per run, requests go through nginx to the API containers.

| Containers | Requests | Failures | Requests/s | Median (ms) | Average (ms) | 95th percentile (ms) | /predict average (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 776 | 0 | 13.5 | 1300 | 1627 | 5600 | 1585 |
| 2 | 1217 | 0 | 20.4 | 550 | 677 | 1500 | 659 |
| 3 | 1510 | 0 | 25.9 | 210 | 267 | 610 | 271 |
