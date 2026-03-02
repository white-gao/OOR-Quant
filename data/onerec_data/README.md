license: apache-2.0
license_link: https://huggingface.co/OpenOneRec/OneRec-8B/blob/main/LICENSE

# OneRec Bench Release Data Documentation


## Files

| File | Description |
|------|-------------|
| `onerec_bench_release.parquet` | Main data file containing multi-domain user behavior data |
| `video_ad_pid2sid.parquet` | Video/Ad item ID to semantic ID mapping |
| `product_pid2sid.parquet` | Goods item ID to semantic ID mapping |
| `pid2caption.parquet` | Item ID to text caption mapping |
| `benchmark_data/` | Evaluation datasets for various recommendation tasks |

---

## Field Descriptions

### Basic Fields

| Field | Type | Description |
|-------|------|-------------|
| `uid` | int | User ID (anonymized) |
| `split` | int | Dataset |

### Short Video Fields


| Field | Type | Description |
|-------|------|-------------|
| `hist_video_pid` | list[int] | User's historical video ID sequence (chronological order, oldest first) |
| `hist_video_longview` | list[int] | Long-view flag for historical videos (0/1, aligned with hist_video_pid) |
| `hist_video_like` | list[int] | Like flag for historical videos (0/1) |
| `hist_video_follow` | list[int] | Follow-author flag for historical videos (0/1) |
| `hist_video_forward` | list[int] | Forward/share flag for historical videos (0/1) |
| `hist_video_not_interested` | list[int] | Not-interested flag for historical videos (0/1) |
| `target_video_pid` | list[int] | Target video ID sequence |
| `target_video_longview` | list[int] | Long-view flag for target videos (ground truth) |
| `target_video_like` | list[int] | Like flag for target videos (ground truth) |
| `target_video_follow` | list[int] | Follow flag for target videos (ground truth) |
| `target_video_forward` | list[int] | Forward flag for target videos (ground truth) |
| `target_video_not_interested` | list[int] | Not-interested flag for target videos (ground truth) |

### Ad Fields


| Field | Type | Description |
|-------|------|-------------|
| `hist_ad_pid` | list[int] | User's historical clicked ad video ID sequence |
| `target_ad_pid` | list[int] | Target ad video ID sequence (ads actually clicked by user) |
| `hist_longview_video_list` | list[int] | Extended long-view video history for ad/product recommendation (longer sequence than `hist_video_pid`, providing richer user interest signals) |

> Note: Ad IDs share the same ID space as video IDs. The `hist_longview_video_list` field contains a longer sequence of videos that the user watched for an extended duration, which serves as auxiliary features for ad and product recommendation tasks.

### Goods Fields


| Field | Type | Description |
|-------|------|-------------|
| `hist_goods_pid` | list[int] | User's historical browsed goods ID sequence|
| `target_goods_pid` | list[int] | Target goods ID sequence |

> Note: Goods IDs use a separate ID space from video/ad IDs. For product recommendation, `hist_longview_video_list` (from Ad Fields) can be used as auxiliary video watching history to capture user interests.

### Interactive Recommendation Fields 


| Field | Type | Description |
|-------|------|-------------|
| `inter_keyword_to_items` | str (JSON) | Keyword to recommended items mapping, format: `{"keyword1": [pid1, pid2, ...], ...}` |
| `inter_user_profile_with_pid` | str | User profile text containing item references in `<photoid\|xxx>` and `<itemid\|xxx>` format |
| `inter_user_profile_with_sid` | str | User profile text containing item references in semantic ID format |

### Recommendation Reason Fields


| Field | Type | Description |
|-------|------|-------------|
| `reco_gsu_caption` | list[str] | Text descriptions of GSU (candidate recall) videos |
| `reco_target_caption` | str | Text description of the target recommended video |
| `reco_cot` | str | Chain-of-Thought reasoning process explaining the recommendation |

---

## Auxiliary Files

### video_ad_pid2sid.parquet

Maps video/ad item IDs to semantic IDs.

| Field | Type | Description |
|-------|------|-------------|
| `pid` | int | Item ID (same ID space as `hist_video_pid`, `target_video_pid`, `hist_ad_pid`, `target_ad_pid`) |
| `sid` | list[int] | Semantic ID array `[s_a, s_b, s_c]`, corresponds to `<s_a_xxx><s_b_xxx><s_c_xxx>` format |

### goods_pid2sid.parquet
Maps goods item IDs to semantic IDs.
| Field | Type | Description |
|-------|------|-------------|
| `pid` | int | Item ID (same ID space as `hist_goods_pid`, `target_goods_pid`) |
| `sid` | list[int] | Semantic ID array `[s_a, s_b, s_c]`, corresponds to `<s_a_xxx><s_b_xxx><s_c_xxx>` format |

### pid2caption.parquet

Maps item IDs to text captions/descriptions.

| Field | Type | Description |
|-------|------|-------------|
| `pid` | int | Item ID |
| `caption` | str | Text description of the item |

---

## Benchmark Data

  The `benchmark_data/` directory contains evaluation datasets for various recommendation tasks. Each parquet file includes the following fields:

  ### Common Fields (All Tasks)

  - **`messages`**: Conversation format for model input (standardized format with `role` and `content` fields)
  - **`metadata`**: JSON string containing ground truth and task-specific information:
    - `answer`: Ground truth answer (all tasks have this field; for fundamental recommendation tasks, this is the semantic ID / itemic token)

  ### Task-Specific Fields

  #### Ad
  - **History fields:**
    - `hist_longview`: List of video IDs from user's long-view history
    - `hist_ad`: List of ad item IDs from user's interaction history
  - **Additional metadata fields:**
    - `answer_pid`: List of ad item IDs (ground truth)

  #### Product
  - **History fields:**
    - `hist_longview`: List of video IDs from user's long-view history
    - `hist_goods`: List of product item IDs from user's purchase/interaction history
  - **Additional metadata fields:**
    - `answer_iid`: List of product item IDs (ground truth)

  #### Video
  - **History fields:**
    - `hist_pid`: List of video IDs from user's interaction history
  - **Additional metadata fields:**
    - `answer_pid`: List of video item IDs (ground truth)

  #### Label Cond
  - **History fields:**
    - `hist_longview`: List of video IDs from long-view history
    - `hist_like`: List of video IDs that user liked
    - `hist_follow`: List of video IDs where user followed the creator
    - `hist_forward`: List of video IDs that user forwarded/shared
    - `hist_not_interested`: List of video IDs marked as not interested
  - **Additional metadata fields:**
    - `answer_pid`: List of video item IDs (ground truth)

  #### Label Pred
  - **History fields:**
    - `hist_longview`: List of video IDs from long-view history
    - `hist_like`: List of video IDs that user liked
    - `hist_follow`: List of video IDs where user followed the creator
    - `hist_forward`: List of video IDs that user forwarded/shared
    - `hist_not_interested`: List of video IDs marked as not interested
    - `candidate_pid`: List of candidate video IDs (can be empty)

  #### Interactive
  - **Additional fields:**
    - `messages_with_pid`: Alternative message format that includes PID information
  - **Additional metadata fields:**
    - `keyword`: Search keyword/query

  #### Rec Reason
  - **Additional fields:**
    - `messages_with_pid`: Alternative message format that includes PID information

  #### Item Understand
  - **Additional metadata fields:**
    - `pid`: video ID


  ### Data Format Notes
  - The `metadata` field is stored as a JSON string and needs to be parsed

## Directory Structure

```
benchmark_data/
├── ad/                    # Ad recommendation evaluation
├── interactive/           # Interactive recommendation evaluation
├── item_understand/       # Item understanding evaluation
├── label_cond/            # Conditional label prediction evaluation
├── label_pred/            # Label prediction evaluation
├── product/               # Product recommendation evaluation
├── rec_reason/            # Recommendation reason generation evaluation
├── video/                 # Video recommendation evaluation
├── sid2iid.json           # Encoded semantic ID (itemic token) to product ID mapping
└── sid2pid.json           # Encoded semantic ID (itemic token) to video/ad ID mapping
```
