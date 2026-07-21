"""
generate_near_miss_negatives.py
=================================
Generates a targeted batch of "near-miss" hard negatives: pairs where
brand, storage, AND color are all IDENTICAL, and only the model/
generation differs (e.g. "iPhone 14 128GB Black" vs "iPhone 15 128GB
Black"). This is the specific failure mode found in production testing
-- the model was scoring these as 100% "Same Product" because every
other signal overlapped, and prior training data didn't isolate this
exact case: it varied storage/color/brand, but never held everything
else fixed while ONLY the model number changed.

Reuses the CATEGORIES data from generate_dataset.py so brand/model
lists stay in one place.

Run:
    python generate_near_miss_negatives.py --n 600 --out data/products_near_miss.csv
"""

import argparse
import csv
import random

from generate_dataset import (
    CATEGORIES,
    _fmt_storage,
    _make_description,
    _make_title,
    _next_id,
    _pick_spec_values,
)

random.seed(44)


def generate_near_miss_pair(category: str):
    """Same brand, same storage, same color -- different model/generation."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    if len(models) < 2:
        return None

    model_a, model_b = random.sample(models, 2)
    storage = random.choice(cat["storages"])
    color, _ = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model_a, storage, color, extra, alt=False)
    title_b = _make_title(brand, model_b, storage, color, extra, alt=False)
    desc_a = _make_description(brand, model_a, _pick_spec_values(category), storage, extra, alt=False)
    desc_b = _make_description(brand, model_b, _pick_spec_values(category), storage, extra, alt=False)

    return (_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0


def generate_near_miss_positive(category: str):
    """Companion positives so the model doesn't overcorrect into 'different
    model number substring anywhere => always different'. Same product,
    same everything, just phrased differently (title alt formatting)."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    storage = random.choice(cat["storages"])
    color_a, color_b = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None
    spec_values = _pick_spec_values(category)

    title_a = _make_title(brand, model, storage, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage, color_b, extra, alt=True)
    desc_a = _make_description(brand, model, spec_values, storage, extra, alt=False)
    desc_b = _make_description(brand, model, spec_values, storage, extra, alt=True)

    return (_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 1


def generate(n_total: int, repeats_per_pair: int = 3):
    """Exhaustively covers every (brand, model_a, model_b) combination at
    least `repeats_per_pair` times with different storage/color, instead
    of relying on random sampling -- guarantees every brand (Pixel,
    Galaxy, iPhone, etc.) gets near-miss coverage, not just whichever
    brands happened to get picked by chance.

    ALSO covers storage-only near-misses: same brand, same model, same
    color -- different storage/RAM/size. This is the same failure mode
    as the model-number case (the model learns "same model name => same
    product" without learning that a differing storage/size still makes
    it a different SKU). For footwear, "storage" is literally the size
    field, so this also resolves same-shoe-different-size."""
    import itertools

    rows = []
    seen = set()

    def _add(side_a, side_b, label):
        title_a, title_b = side_a[1], side_b[1]
        if title_a.lower() == title_b.lower():
            return False
        key, key_rev = (title_a.lower(), title_b.lower()), (title_b.lower(), title_a.lower())
        if key in seen or key_rev in seen:
            return False
        seen.add(key)
        rows.append((*side_a, *side_b, label))
        return True

    for category, data in CATEGORIES.items():
        for brand, models in data["brands_models"]:
            # ---- Model near-misses (existing) ----
            if len(models) >= 2:
                for model_a, model_b in itertools.combinations(models, 2):
                    for _ in range(repeats_per_pair):
                        storage = random.choice(data["storages"])
                        color, _ = random.choice(data["colors"])
                        extra = random.choice(data.get("extra", [None])) if "extra" in data else None

                        title_a = _make_title(brand, model_a, storage, color, extra, alt=False)
                        title_b = _make_title(brand, model_b, storage, color, extra, alt=False)
                        desc_a = _make_description(brand, model_a, _pick_spec_values(category), storage, extra, alt=False)
                        desc_b = _make_description(brand, model_b, _pick_spec_values(category), storage, extra, alt=False)
                        _add((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)

                        same_model = random.choice([model_a, model_b])
                        color_a2, color_b2 = random.choice(data["colors"])
                        spec_values = _pick_spec_values(category)
                        pt_a = _make_title(brand, same_model, storage, color_a2, extra, alt=False)
                        pt_b = _make_title(brand, same_model, storage, color_b2, extra, alt=True)
                        pd_a = _make_description(brand, same_model, spec_values, storage, extra, alt=False)
                        pd_b = _make_description(brand, same_model, spec_values, storage, extra, alt=True)
                        _add((_next_id(), pt_a, brand, pd_a), (_next_id(), pt_b, brand, pd_b), 1)

            # ---- Storage/size near-misses: same model, different storage ----
            storages = data.get("storages", [])
            valid_storages = [s for s in storages if s is not None]
            if len(valid_storages) >= 2:
                for model in models:
                    for storage_a, storage_b in itertools.combinations(valid_storages, 2):
                        for _ in range(repeats_per_pair):
                            color, _ = random.choice(data["colors"])
                            extra = random.choice(data.get("extra", [None])) if "extra" in data else None

                            title_a = _make_title(brand, model, storage_a, color, extra, alt=False)
                            title_b = _make_title(brand, model, storage_b, color, extra, alt=False)
                            desc_a = _make_description(brand, model, _pick_spec_values(category), storage_a, extra, alt=False)
                            desc_b = _make_description(brand, model, _pick_spec_values(category), storage_b, extra, alt=False)
                            _add((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)

                            # Companion positive: same storage, reworded.
                            same_storage = random.choice([storage_a, storage_b])
                            spec_values = _pick_spec_values(category)
                            pt_a = _make_title(brand, model, same_storage, color, extra, alt=False)
                            pt_b = _make_title(brand, model, same_storage, color, extra, alt=True)
                            pd_a = _make_description(brand, model, spec_values, same_storage, extra, alt=False)
                            pd_b = _make_description(brand, model, spec_values, same_storage, extra, alt=True)
                            _add((_next_id(), pt_a, brand, pd_a), (_next_id(), pt_b, brand, pd_b), 1)

            # ---- Color-only near-misses: same model, same storage, DIFFERENT actual color ----
            # (not the two-spellings-of-the-same-color pairs used for positives --
            # a genuinely different color entry from the palette)
            colors_list = data.get("colors", [])
            if len(colors_list) >= 2:
                for model in models:
                    for (color_a, _), (color_b, _) in itertools.combinations(colors_list, 2):
                        if color_a.lower() == color_b.lower():
                            continue
                        for _ in range(max(1, repeats_per_pair // 2)):
                            storage = random.choice(data["storages"])
                            extra = random.choice(data.get("extra", [None])) if "extra" in data else None
                            title_a = _make_title(brand, model, storage, color_a, extra, alt=False)
                            title_b = _make_title(brand, model, storage, color_b, extra, alt=False)
                            desc_a = _make_description(brand, model, _pick_spec_values(category), storage, extra, alt=False)
                            desc_b = _make_description(brand, model, _pick_spec_values(category), storage, extra, alt=False)
                            _add((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)

                            # Companion positive: same color, same everything, just
                            # reworded -- keeps the model from overcorrecting into
                            # "any color mention differing at all => different".
                            same_color = random.choice([color_a, color_b])
                            spec_values = _pick_spec_values(category)
                            pt_a = _make_title(brand, model, storage, same_color, extra, alt=False)
                            pt_b = _make_title(brand, model, storage, same_color, extra, alt=True)
                            pd_a = _make_description(brand, model, spec_values, storage, extra, alt=False)
                            pd_b = _make_description(brand, model, spec_values, storage, extra, alt=True)
                            _add((_next_id(), pt_a, brand, pd_a), (_next_id(), pt_b, brand, pd_b), 1)

            # ---- Extra-spec-only near-misses: same model/storage/color, DIFFERENT
            # extra spec (e.g. CPU: i5 vs i7, chip: M2 Pro vs M3 Pro) ----
            extras = [e for e in data.get("extra", []) if e is not None]
            if len(extras) >= 2:
                for model in models:
                    for extra_a, extra_b in itertools.combinations(extras, 2):
                        for _ in range(max(1, repeats_per_pair // 2)):
                            storage = random.choice(data["storages"])
                            color, _ = random.choice(data["colors"])
                            title_a = _make_title(brand, model, storage, color, extra_a, alt=False)
                            title_b = _make_title(brand, model, storage, color, extra_b, alt=False)
                            desc_a = _make_description(brand, model, _pick_spec_values(category), storage, extra_a, alt=False)
                            desc_b = _make_description(brand, model, _pick_spec_values(category), storage, extra_b, alt=False)
                            _add((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)

                            # Companion positive: same extra spec, reworded.
                            same_extra = random.choice([extra_a, extra_b])
                            spec_values = _pick_spec_values(category)
                            pt_a = _make_title(brand, model, storage, color, same_extra, alt=False)
                            pt_b = _make_title(brand, model, storage, color, same_extra, alt=True)
                            pd_a = _make_description(brand, model, spec_values, storage, same_extra, alt=False)
                            pd_b = _make_description(brand, model, spec_values, storage, same_extra, alt=True)
                            _add((_next_id(), pt_a, brand, pd_a), (_next_id(), pt_b, brand, pd_b), 1)

    random.shuffle(rows)
    n_pos = sum(1 for r in rows if r[-1] == 1)
    print(f"Generated {len(rows)} rows (model + storage/size near-misses) covering every brand "
          f"at least {repeats_per_pair}x (positives={n_pos}, negatives={len(rows) - n_pos})")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3,
                         help="how many times to regenerate each brand's model-pair "
                              "combination with different storage/color (more = more diversity)")
    parser.add_argument("--out", default="data/products_near_miss.csv")
    args = parser.parse_args()

    rows = generate(n_total=0, repeats_per_pair=args.repeats)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product1_id", "product1_title", "product1_brand", "product1_description",
            "product2_id", "product2_title", "product2_brand", "product2_description",
            "label",
        ])
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r[-1] == 1)
    n_neg = len(rows) - n_pos
    print(f"Wrote {len(rows)} rows to {args.out} (positives={n_pos}, near-miss negatives={n_neg})")


if __name__ == "__main__":
    main()