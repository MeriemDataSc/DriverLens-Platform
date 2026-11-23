# `our data/` — raw comment exports (not versioned)

Place the YouTube comment exports here (`.csv`, `.xlsx` or `.xlsm`). The folder content is ignored by Git
because the exports contain personal data (author names, channel IDs, avatar URLs).

## Files used in the project

| File (one export per video) | Car model | Format |
|---|---|---|
| `Renault 5 E-Tech electrique_1.csv`, `Renault 5 E-Tech électrique_2.csv` | Renault 5 E-Tech | CSV export |
| `Renault Mégane E-Tech Electric_1…4.xlsx` | Mégane E-Tech | Excel export |
| `Renault Scénic E-Tech Electric_1…5.xlsx` | Scénic E-Tech | Excel export |
| `Renault R4 E-Tech électrique.xlsx` | Renault 4 E-Tech | Excel export |

## Accepted columns

`utils/data_loader.py` recognises the two export layouts automatically (column names are matched through aliases):

- **CSV layout**: `Author, Comment, Published At, Edited At, Likes, Replies, Reply To (Comment ID), Comment ID, Comment Link, Author Channel, Author Avatar, Video ID`
- **Excel layout**: `id, reply_to, youtube_comment_id, author, text, published_at, likes, reply_count, is_creator, is_verified, author_channel_id, avatar_url`

The car model is inferred from the file name, the video title or the comments themselves; it can be forced in
`config.py → RUN_CONFIG["file_car_models"]`.
