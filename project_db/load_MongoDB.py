import os
import csv
from typing import Dict, List
from pymongo import UpdateOne
from mongoDB_tools import MongoTools 

class LoadMongoDB:
    """
    MongoDB loader for anime dataset.
    """

    def __init__(
        self,
        mongo_tools: MongoTools, 
        batch_size: int = 5000,
        progress_interval: int = 50_000,
    ):
        self.tools = mongo_tools 
        self.batch = batch_size
        self.progress_every = progress_interval

 
    # 1) FULL ANIME LOADER (03_load_mongo_anime.py)
    def load_animes(self, csv_syn_path: str, csv_meta_path: str, collection_name: str):
        tools = self.tools
        col = tools.get_mongo_collection(collection_name) # Access managed collection

        # Load meta map
        meta_map = self._load_meta_map(csv_meta_path)

        ops = []
        total = upserts = 0

        with open(csv_syn_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)

            for row in rdr:
                r = tools.lower_keys(row)

                mal_id = tools.to_int(r.get("mal_id"))
                if not mal_id:
                    continue

                # Main fields
                name = tools.clean_str(r.get("name")) \
                       or meta_map.get(mal_id, {}).get("name")

                synopsis = tools.clean_str(r.get("synopsis"))

                episodes = (
                    tools.to_int(r.get("episodes"))
                    if r.get("episodes") is not None
                    else meta_map.get(mal_id, {}).get("episodes")
                )

                type_ = tools.clean_str(r.get("type")) or meta_map.get(mal_id, {}).get("type")
                aired = tools.clean_str(r.get("aired")) or meta_map.get(mal_id, {}).get("aired")
                premiered = tools.clean_str(r.get("premiered")) or meta_map.get(mal_id, {}).get("premiered")
                duration = tools.clean_str(r.get("duration")) or meta_map.get(mal_id, {}).get("duration")
                age_rating = tools.clean_str(r.get("rating")) or meta_map.get(mal_id, {}).get("age_rating")

                # multi-value arrays
                genres = tools.split_list(r.get("genres")) \
                         or meta_map.get(mal_id, {}).get("genres") \
                         or []

                studios = tools.split_list(r.get("studios")) \
                          or meta_map.get(mal_id, {}).get("studios") \
                          or []

                producers = tools.split_list(r.get("producers")) \
                            or meta_map.get(mal_id, {}).get("producers") \
                            or []

                licensors = tools.split_list(r.get("licensors")) \
                            or meta_map.get(mal_id, {}).get("licensors") \
                            or []

                # stats
                stats = {
                    "score": tools.to_float2(r.get("score"))
                    if "score" in r else meta_map.get(mal_id, {}).get("stats", {}).get("score"),

                    "rank": tools.to_int(r.get("rank"))
                    if "rank" in r else meta_map.get(mal_id, {}).get("stats", {}).get("rank"),

                    "popularity": tools.to_int(r.get("popularity"))
                    if "popularity" in r else meta_map.get(mal_id, {}).get("stats", {}).get("popularity"),

                    "members": tools.to_int(r.get("members"))
                    if "members" in r else meta_map.get(mal_id, {}).get("stats", {}).get("members"),

                    "favorites": tools.to_int(r.get("favorites"))
                    if "favorites" in r else meta_map.get(mal_id, {}).get("stats", {}).get("favorites"),
                }

                stats = {k: v for k, v in stats.items() if v is not None}
                if not stats:
                    stats = None

                # Document
                doc = {
                    "_id": mal_id,
                    "name": name,
                    "synopsis": synopsis,
                    "type": type_,
                    "episodes": episodes,
                    "aired": aired,
                    "premiered": premiered,
                    "duration": duration,
                    "age_rating": age_rating,
                    "genres": genres,
                    "studios": studios,
                    "producers": producers,
                    "licensors": licensors,
                }
                if stats:
                    doc["stats"] = stats

                # clean Nones
                doc = {k: v for k, v in doc.items() if v is not None}

                ops.append(UpdateOne({"_id": mal_id}, {"$set": doc}, upsert=True))
                total += 1

                if len(ops) >= self.batch:
                    res = col.bulk_write(ops, ordered=False)
                    upserts += (res.upserted_count or 0)
                    ops.clear()

                if total % self.progress_every == 0:
                    print(f"processed={total:,}, upserts≈{upserts:,}")

        # final flush
        if ops:
            res = col.bulk_write(ops, ordered=False)
            upserts += (res.upserted_count or 0)

        print(f"Done loading animes. processed={total:,}, upserts≈{upserts:,}")
        print("collection size:", col.estimated_document_count())


    # Helper: load meta map 
    def _load_meta_map(self, csv_meta_path: str) -> Dict[int, dict]:
        tools = self.tools
        if not os.path.exists(csv_meta_path):
            print("Note: anime.csv not found -> meta enrichment skipped.")
            return {}

        meta = {}
        with open(csv_meta_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                r = tools.lower_keys(row)
                mid = tools.to_int(r.get("mal_id"))
                if not mid:
                    continue

                meta[mid] = {
                    "name": tools.clean_str(r.get("name")),
                    "episodes": tools.to_int(r.get("episodes")),
                    "type": tools.clean_str(r.get("type")),
                    "aired": tools.clean_str(r.get("aired")),
                    "premiered": tools.clean_str(r.get("premiered")),
                    "duration": tools.clean_str(r.get("duration")),
                    "age_rating": tools.clean_str(r.get("rating")),
                    "genres": tools.split_list(r.get("genres")),
                    "studios": tools.split_list(r.get("studios")),
                    "producers": tools.split_list(r.get("producers")),
                    "licensors": tools.split_list(r.get("licensors")),
                    "stats": {
                        "score": tools.to_float2(r.get("score")),
                        "rank": tools.to_int(r.get("rank")),
                        "popularity": tools.to_int(r.get("popularity")),
                        "members": tools.to_int(r.get("members")),
                        "favorites": tools.to_int(r.get("favorites")),
                    },
                }
        print(f"Loaded meta map: {len(meta):,} rows")
        return meta


    # 2) Synopsis loader
    def load_synopsis(
        self,
        csv_path: str,
        collection_name: str,
        overwrite_existing: bool = True,
        upsert_missing: bool = True,
    ):
        tools = self.tools
        col = tools.get_mongo_collection(collection_name)

        ops = []
        updated = upserts = skipped = total_rows = 0

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        with open(csv_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)

            for row in rdr:
                r = tools.lower_keys(row)

                mal_id = tools.to_int(r.get("mal_id") or r.get("id"))
                synopsis = tools.clean_str(
                    r.get("synopsis")
                    or r.get("synopsys")
                    or r.get("synopsis_text")
                    or r.get("sypnopsis")
                )

                if not mal_id:
                    continue
                if not synopsis:
                    skipped += 1
                    continue

                # Filter for upsert
                if overwrite_existing:
                    filt = {"_id": mal_id}
                else:
                    filt = {"_id": mal_id, "$or": [
                        {"synopsis": {"$exists": False}},
                        {"synopsis": ""}
                    ]}

                ops.append(UpdateOne(filt, {"$set": {"synopsis": synopsis}}, upsert=upsert_missing))
                total_rows += 1

                # batch write
                if len(ops) >= self.batch:
                    res = col.bulk_write(ops, ordered=False)
                    updated += (res.modified_count or 0)
                    upserts += (res.upserted_count or 0)
                    ops.clear()

        # flush
        if ops:
            res = col.bulk_write(ops, ordered=False)
            updated += (res.modified_count or 0)
            upserts += (res.upserted_count or 0)

        print(f"Done loading synopsis. read={total_rows:,}, updated={updated:,}, upserts={upserts:,}, skipped={skipped:,}")
        print("Docs with synopsis:",
              col.count_documents({"synopsis": {"$exists": True, "$ne": ""}}),
              "/",
              col.estimated_document_count())

    # 3) Alternative names loader
    def load_alt_names(
        self,
        csv_syn_path: str,
        csv_meta_path: str,
        alt_collection: str,
        anime_collection: str,
    ):
        tools = self.tools

        alt_col = tools.get_mongo_collection(alt_collection)
        anime_col = tools.get_mongo_collection(anime_collection)

        # Load maps
        map_syn = self._load_alt_name_map(csv_syn_path)
        map_meta = self._load_alt_name_map(csv_meta_path)

        all_ids = set(map_syn.keys()) | set(map_meta.keys())

        ops_alt = []
        total = upserts = 0

        for mal_id in all_ids:
            a = map_syn.get(mal_id)
            b = map_meta.get(mal_id)

            merged = self._merge_alt_records(a, b)

            # Clean empty
            merged["synonyms"] = [s for s in merged.get("synonyms", []) if s]

            if (
                not merged.get("english_name")
                and not merged.get("japanese_name")
                and not merged["synonyms"]
            ):
                continue

            doc = {"_id": mal_id}
            if merged.get("english_name"):
                doc["english_name"] = merged["english_name"]
            if merged.get("japanese_name"):
                doc["japanese_name"] = merged["japanese_name"]
            if merged["synonyms"]:
                doc["synonyms"] = merged["synonyms"]

            ops_alt.append(UpdateOne({"_id": mal_id}, {"$set": doc}, upsert=True))
            total += 1

            if len(ops_alt) >= self.batch:
                res = alt_col.bulk_write(ops_alt, ordered=False)
                upserts += (res.upserted_count or 0)
                ops_alt.clear()

            if total % self.progress_every == 0:
                print(f"alternative names processed={total:,} upserts≈{upserts:,}")

        # flush
        if ops_alt:
            res = alt_col.bulk_write(ops_alt, ordered=False)
            upserts += (res.upserted_count or 0)

        print(f"Done loading alternative names. processed={total:,}, upserts≈{upserts:,}")
        print("Alternative names collection size:", alt_col.estimated_document_count())

    # Helter for alt names (No change)
    def _load_alt_name_map(self, csv_path: str) -> Dict[int, dict]:
        tools = self.tools

        if not os.path.exists(csv_path):
            return {}

        out = {}

        with open(csv_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)

            for row in rdr:
                r = tools.lower_keys(row)

                mal_id = tools.get_first_nonempty(r, ["mal_id", "id", "malid"])
                if not mal_id or not mal_id.isdigit():
                    continue

                mid = int(mal_id)

                english = tools.get_first_nonempty(
                    r,
                    ["english name", "english_name", "english", "title_english"],
                )
                japanese = tools.get_first_nonempty(
                    r,
                    ["japanese name", "japanese_name", "japanese", "title_japanese"],
                )

                syn_raw = tools.get_first_nonempty(
                    r,
                    ["synonyms", "other name", "other names", "other_names", "title_synonyms"],
                )
                synonyms = tools.split_list(syn_raw)

                out[mid] = {
                    "english_name": english,
                    "japanese_name": japanese,
                    "synonyms": synonyms,
                }

        return out

    def _merge_alt_records(self, a: dict, b: dict) -> dict:
        tools = self.tools

        if not a:
            return b or {"english_name": None, "japanese_name": None, "synonyms": []}
        if not b:
            return a or {"english_name": None, "japanese_name": None, "synonyms": []}

        english = a.get("english_name") or b.get("english_name")
        japanese = a.get("japanese_name") or b.get("japanese_name")

        syn = tools.merge_synonyms(
            a.get("synonyms") or [],
            b.get("synonyms") or []
        )

        return {"english_name": english, "japanese_name": japanese, "synonyms": syn}