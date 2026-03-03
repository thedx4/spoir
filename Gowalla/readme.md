# Gowalla Dataset

## Overview

This dataset contains user check-in activity and friendship relations derived from the Gowalla location-based social network.

The dataset is composed of two CSV files:

1. **gowalla_checkins.csv**
2. **gowalla_friendship.csv**

**Download Link:**
https://drive.google.com/drive/folders/1DkmJN5B1jzQidu73BKbDv1tJ1IZC57rQ?usp=sharing

---

## File Descriptions

### 1. gowalla_checkins.csv

* **Format:** Comma-separated values (CSV)
* **Total Entries:** 36,001,959

#### Columns

| Column Name | Description                            |
| ----------- | -------------------------------------- |
| `userid`    | Unique numeric identifier of the user  |
| `placeid`   | Unique numeric identifier of the place |
| `datetime`  | Timestamp of the check-in (GMT)        |

* `userid` and `placeid` are numeric IDs.
* `datetime` is recorded in **GMT** timezone.

Each row represents a single check-in event performed by a user at a specific place and time.

---

### 2. gowalla_friendship.csv

* **Format:** Comma-separated values (CSV)
* **Total Entries:** 4,418,339

#### Columns

| Column Name | Description                                     |
| ----------- | ----------------------------------------------- |
| `userid1`   | Unique numeric identifier of a user             |
| `userid2`   | Unique numeric identifier of the connected user |

Each row represents a friendship relationship between two users.

---

## Dataset Statistics

| Component  | Format | Entries    |
| ---------- | ------ | ---------- |
| Check-ins  | CSV    | 36,001,959 |
| Friendship | CSV    | 4,418,339  |
