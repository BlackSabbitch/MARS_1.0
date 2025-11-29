import json
import os
import pycountry
from pathlib import Path

class LocationCleaner:
    def __init__(self):
        self.data_path = Path(__file__).parent.parent.joinpath("data").joinpath("data_cda")

        self.known_locations = self._load_json("known_locations")
        self.us_states = self._load_json("us_states.json")
        self.unknown_locations = self._load_json("unknown_locations")

    def _load_json(self, filename):
        path = os.path.join(self.data_path, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else: raise ValueError(f"[ERROR] No File {filename}!!!!")

    def clean(self, location):
        if not isinstance(location, str) or location.strip() == "":
            return "Unknown"

        loc = location.strip()

    
        if loc in self.known_locations:
            return self.known_locations[loc]
        if loc in self.unknown_locations:
            return self.unknown_locations[loc]

    
        parts = [p.strip() for p in loc.split(",")]

    
        for part in reversed(parts):
        
            key_lower = part.lower()
            for k, v in self.known_locations.items():
                if k.lower() == key_lower:
                    return v
            for k, v in self.unknown_locations.items():
                if k.lower() == key_lower:
                    return v
            if key_lower in (s.lower() for s in self.us_states):
                return "United States"

        
            try:
                match = pycountry.countries.lookup(part)
                return getattr(match, "common_name", match.name)
            except LookupError:
                pass

        
            words = part.split()
            for w in reversed(words):
                try:
                    match = pycountry.countries.lookup(w)
                    return getattr(match, "common_name", match.name)
                except LookupError:
                    pass

        return loc.title()
    
    def map_as_unknown(self, df, column="country", min_count=25):
        counts = df[column].value_counts()
        df[column] = df[column].apply(lambda x: x if counts.get(x,0) >= min_count else "Unknown")
        return df
        