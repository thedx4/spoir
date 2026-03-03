
# Foursquare Dataset

## Overview

This dataset contains user check-in activity and friendship relations derived from Foursquare data.
It is divided into two main components:

1. **Foursquare Check-ins** (split into 4 parts)
2. **Foursquare Friendship Network**

---

## Files Included

### 1. Foursquare Check-ins

The check-in data is split into four parts:

* `Foursquare_Checkins_1.txt`
* `Foursquare_Checkins_2.txt`
* `Foursquare_Checkins_3.txt`
* `Foursquare_Checkins_4.txt`

> These files should be combined to reconstruct the full dataset.

When combined, the dataset contains **483,813 total entries**

### Columns

The combined file contains the following columns:

| Column Name     | Description                         |
| --------------- | ----------------------------------- |
| `userID`        | Unique identifier of the user       |
| `Time(GMT)`     | Timestamp of the check-in (GMT)     |
| `VenueId`       | Unique identifier of the venue      |
| `VenueName`     | Name of the venue                   |
| `VenueLocation` | Structured location data            |
| `VenueCategory` | Category or categories of the venue |

### VenueLocation Format

`VenueLocation` is stored as a structured value containing:

```
{latitude, longitude, city, state, country}
```

Example:

```
{37.8246278550847,-122.29038447609702,Oakland,CA,United States}
```

---

### 2. Foursquare Friendship

File:

* `Foursquare_Friendship.csv`

This file contains user friendship relationships.

* **32,511 total entries**
* CSV format (comma-separated)

### Columns

| Column Name | Description                     |
| ----------- | ------------------------------- |
| `userID`    | Unique identifier of the user   |
| `friendID`  | Unique identifier of the friend |

Each row represents a friendship connection between two users.
