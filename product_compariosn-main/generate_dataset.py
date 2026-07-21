"""
generate_dataset.py
====================
Generates a synthetic-but-realistic product-matching dataset matching
the project's title-only schema: product1, product2, label.

Design goals (directly addressing the small-dataset generalization gap):
  - Many categories/brands, not just phones -> forces the model to learn
    a general "same entity, different phrasing" pattern instead of
    memorizing a handful of brand names.
  - POSITIVE pairs: same canonical product, rendered two different ways
    (the way two different e-commerce sites would phrase the same
    listing: unit spacing, parentheses, word order, abbreviations,
    color spelling).
  - HARD NEGATIVE pairs: same brand + same category, but a genuinely
    different variant (different storage/RAM/color/model number) --
    this is the case a shallow "same brand => match" heuristic gets
    wrong, and is exactly what was missing from the original 218-row
    dataset.
  - EASY NEGATIVE pairs: different category/brand entirely.

Run:
    python generate_dataset.py --n 1200 --out data/products_1000.csv
"""

import argparse
import csv
import itertools
import random


random.seed(42)


# --------------------------------------------------------------------------
# Category definitions: brand -> model families -> variant attributes
# --------------------------------------------------------------------------
CATEGORIES = {
    "smartphone": {
        "brands_models": [
            ("Apple", ["iPhone 13", "iPhone 14", "iPhone 15", "iPhone 15 Pro", "iPhone 16", "iPhone 16 Pro"]),
            ("Samsung", ["Galaxy S23", "Galaxy S23 Ultra", "Galaxy S24", "Galaxy S24 Ultra", "Galaxy A54", "Galaxy Z Flip5"]),
            ("OnePlus", ["11", "12", "Nord 3", "12R"]),
            ("Xiaomi", ["Redmi Note 13 Pro", "Redmi Note 12", "Mi 13", "Poco X6 Pro", "Xiaomi 14", "Xiaomi 14 Ultra"]),
            ("Google", ["Pixel 7", "Pixel 8", "Pixel 8 Pro", "Pixel 8a"]),
            ("Vivo", ["V29", "V27 Pro", "X90"]),
            ("Oppo", ["Reno 10", "Reno 11 Pro", "F25 Pro"]),
            ("Realme", ["11 Pro", "GT 5", "Narzo 60"]),
            ("Motorola", ["Edge 40", "Edge 50 Pro", "Moto G84"]),
            ("Nothing", ["Phone 1", "Phone 2", "Phone 2a"]),
        ],
        "storages": ["64GB", "128GB", "256GB", "512GB"],
        "colors": [("Black", "Black"), ("Titanium Gray", "Titanium Grey"), ("Blue", "Blue"),
                   ("Green", "Green"), ("White", "White"), ("Midnight", "Midnight"),
                   ("Desert Titanium", "Desert Titanium"), ("Phantom Black", "Phantom Black")],
        # Condition matters as much as spec for phones (New vs Renewed
        # should be a different listing) -- found as a gap in testing.
        "extra": ["New", "Renewed", "Refurbished", "Open Box", None],
    },
    "laptop": {
        "brands_models": [
            ("HP", ["Victus", "Pavilion 15", "Envy x360", "Omen 16"]),
            ("Dell", ["Inspiron 15", "XPS 13", "Vostro 14", "Alienware m16"]),
            ("Lenovo", ["ThinkPad E14", "IdeaPad Slim 5", "Legion 5", "Yoga Slim 7"]),
            ("Asus", ["Vivobook 15", "ROG Strix G16", "Zenbook 14", "TUF Gaming A15"]),
            ("Acer", ["Aspire 7", "Nitro 5", "Swift 3", "Predator Helios"]),
            ("Apple", ["MacBook Air M2", "MacBook Air M3", "MacBook Pro 14 M3"]),
            ("MSI", ["Katana 15", "Modern 14", "Cyborg 15"]),
        ],
        "storages": ["256GB SSD", "512GB SSD", "1TB SSD"],
        "colors": [("Silver", "Silver"), ("Black", "Black"), ("Space Gray", "Space Grey"), ("Blue", "Blue")],
        "extra": ["Ryzen 7 RTX 4060", "Core i5 RTX 3050", "Ryzen 5", "Core i7 16GB RAM", "Core i9 RTX 4070",
                   "Windows 11", "Ubuntu Linux", "Touch Display", "Non-Touch Display"],
    },
    "earbuds": {
        "brands_models": [
            ("Boat", ["Airdopes 311 Pro", "Airdopes 141", "Airdopes 441", "Rockerz 255"]),
            ("JBL", ["Tune 230NC", "Wave Buds", "Tour Pro 2", "Live Pro 2"]),
            ("Sony", ["WF-1000XM4", "WF-1000XM5", "WF-C700N"]),
            ("Apple", ["AirPods Pro 2", "AirPods 3rd Gen", "AirPods 4"]),
            ("Samsung", ["Galaxy Buds2 Pro", "Galaxy Buds FE", "Galaxy Buds3"]),
            ("Noise", ["Buds VS104", "Air Buds", "Buds Prima"]),
            ("Realme", ["Buds Air 5", "Buds T300", "Buds Q3s"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("White", "White"), ("Blue", "Blue"), ("Teal", "Teal")],
    },
    "smartwatch": {
        "brands_models": [
            ("Apple", ["Watch Series 9", "Watch SE", "Watch Ultra 2"]),
            ("Samsung", ["Galaxy Watch6", "Galaxy Watch6 Classic", "Galaxy Watch FE"]),
            ("Noise", ["ColorFit Pulse 2", "ColorFit Ultra 3", "ColorFit Icon 2"]),
            ("Boat", ["Wave Neo", "Xtend", "Storm Pro"]),
            ("Fire-Boltt", ["Phoenix Pro", "Ninja Call Pro", "Talk 2"]),
            ("Fitbit", ["Versa 4", "Sense 2", "Charge 6"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("Rose Gold", "Rose Gold")],
    },
    "television": {
        "brands_models": [
            ("Samsung", ["Crystal 4K UHD", "The Frame QLED", "Neo QLED 4K"]),
            ("LG", ["UQ7500 4K", "OLED C3", "NanoCell 4K"]),
            ("Sony", ["Bravia X75K", "Bravia X90L"]),
            ("Mi", ["X Pro 4K", "5A Pro"]),
            ("OnePlus", ["Y1S Pro", "U1S"]),
        ],
        "storages": ["43 inch", "50 inch", "55 inch", "65 inch"],
        "colors": [("Black", "Black")],
    },
    "footwear": {
        "brands_models": [
            ("Nike", ["Air Max 270", "Revolution 6", "Air Force 1", "Pegasus 40"]),
            ("Adidas", ["Ultraboost 22", "Duramo SL", "Superstar", "Runfalcon 3"]),
            ("Puma", ["Softride Rift", "Smash v2", "Anzarun Lite"]),
            ("Reebok", ["Classic Leather", "Energen Lux", "Flexagon Force"]),
            ("Skechers", ["Go Walk 6", "Summits", "Flex Advantage"]),
        ],
        "storages": ["UK 6", "UK 7", "UK 8", "UK 9", "UK 10"],
        "colors": [("Black/White", "Black/White"), ("Triple Black", "Triple Black"), ("Grey", "Gray"), ("Blue", "Blue")],
    },
    "kitchen_appliance": {
        "brands_models": [
            ("Prestige", ["Mixer Grinder 750W", "Induction Cooktop", "Electric Kettle 1.5L"]),
            ("Philips", ["Air Fryer HD9252", "Mixer Grinder HL7756", "Juicer HR1832"]),
            ("Bajaj", ["Majesty Toaster", "Mixer Grinder Rex", "Induction Cooktop 1400W"]),
            ("Havells", ["Toaster ST-11", "Electric Kettle Aquis"]),
            ("Butterfly", ["Mixer Grinder Rocket", "Induction Cooktop Rapid"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("Red", "Red"), ("White", "White"), ("Steel", "Steel")],
    },
    "gaming_console": {
        "brands_models": [
            ("Sony", ["PlayStation 5", "PlayStation 5 Pro", "PlayStation 4"]),
            ("Microsoft", ["Xbox Series X", "Xbox Series S"]),
            ("Nintendo", ["Switch OLED", "Switch Lite", "Switch 2"]),
        ],
        # "storages" doubles as both capacity AND edition here -- gaming
        # consoles commonly vary by both, and each is a genuine near-miss
        # trap ("Disc Edition" vs "Digital Edition", "1TB" vs "2TB").
        "storages": ["512GB Disc Edition", "512GB Digital Edition", "1TB", "2TB"],
        "colors": [("White", "White"), ("Black", "Black")],
    },
    "camera": {
        "brands_models": [
            ("Canon", ["EOS R50", "EOS R10", "EOS R6 Mark II", "EOS 90D"]),
            ("Sony", ["Alpha a7 IV", "Alpha a6400", "ZV-E10"]),
            ("Nikon", ["Z50", "Z6 III", "D7500"]),
            ("Fujifilm", ["X-T5", "X-S20"]),
        ],
        "storages": ["Body Only", "with 18-55mm Lens", "with 24-70mm Lens"],
        "colors": [("Black", "Black")],
        "extra": ["Mirrorless Camera", "DSLR Camera", "APS-C Sensor", "Full Frame Sensor"],
    },
    "monitor": {
        "brands_models": [
            ("Dell", ["UltraSharp U2723QE", "UltraSharp U3223QE", "S2721DGF"]),
            ("LG", ["27GP850", "34WP65C", "UltraGear 27GN800"]),
            ("Samsung", ["Odyssey G7", "ViewFinity S8", "Odyssey Neo G9"]),
            ("ASUS", ["ProArt PA278QV", "TUF Gaming VG27AQ"]),
        ],
        "storages": ["24 inch", "27 inch", "32 inch", "34 inch"],
        "colors": [("Black", "Black"), ("Silver", "Silver")],
    },
    "storage_device": {
        "brands_models": [
            ("Samsung", ["990 PRO", "970 EVO Plus", "T7 Shield"]),
            ("WD", ["Black SN850X", "Blue SN580", "My Passport"]),
            ("Crucial", ["P5 Plus", "MX500", "X6 Portable"]),
            ("SanDisk", ["Extreme Portable", "Extreme Pro"]),
        ],
        "storages": ["500GB", "1TB", "2TB", "4TB"],
        "colors": [("Black", "Black")],
        "extra": ["NVMe SSD", "PCIe 4.0", "Portable SSD", "SATA SSD"],
    },
    "vacuum": {
        "brands_models": [
            ("Dyson", ["V15 Detect", "V12 Detect Slim", "V8", "Ball Animal 3"]),
            ("Shark", ["Navigator Lift-Away", "Vertex Pro", "Stratos"]),
            ("iRobot", ["Roomba j7+", "Roomba 694", "Roomba Combo j9+"]),
            ("Eureka", ["NEC122", "PowerSpeed"]),
        ],
        "storages": [None],
        "colors": [("Yellow/Nickel", "Yellow/Nickel"), ("Black", "Black"), ("Silver", "Silver")],
        "extra": ["Cordless Vacuum", "Robot Vacuum", "Bagless Upright",
                   "with crevice tool", "with pet hair tool", "with extra filter"],
    },
    "book": {
        "brands_models": [
            ("George Orwell", ["1984", "Animal Farm"]),
            ("J.R.R. Tolkien", ["The Hobbit", "The Fellowship of the Ring"]),
            ("Jane Austen", ["Pride and Prejudice", "Sense and Sensibility"]),
            ("Suzanne Collins", ["The Hunger Games", "Catching Fire"]),
        ],
        "storages": [None],
        "colors": [("Paperback", "Paperback"), ("Hardcover", "Hardcover")],
        # "extra" here doubles as edition -- the exact gap the regression suite found.
        "extra": ["1st Edition", "2nd Edition", "3rd Edition", "Anniversary Edition",
                   "Illustrated Edition", None],
    },
    "video_game": {
        "brands_models": [
            ("EA Sports", ["FIFA 23", "Madden NFL 24"]),
            ("CD Projekt Red", ["The Witcher 3", "Cyberpunk 2077"]),
            ("Rockstar Games", ["Grand Theft Auto V", "Red Dead Redemption 2"]),
            ("Nintendo", ["The Legend of Zelda: Tears of the Kingdom", "Super Mario Odyssey"]),
        ],
        "storages": [None],
        "colors": [("Standard", "Standard")],
        # platform AND edition both live here -- both were found as gaps.
        "extra": ["PS5", "Xbox Series X", "Nintendo Switch", "PC",
                   "Standard Edition", "Game of the Year Edition", "Deluxe Edition"],
    },
    "networking": {
        "brands_models": [
            ("TP-Link", ["Archer AX50", "Archer AX21", "Deco X60"]),
            ("Netgear", ["Nighthawk AX12", "Orbi RBK852"]),
            ("ASUS", ["RT-AX88U", "ZenWiFi AX"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("White", "White")],
        "extra": ["Wi-Fi 6", "Dual-Band", "v1", "v2", "v3"],
    },
}


TITLE_TEMPLATES = [
    "{brand} {model} {storage} {color}",
    "{brand} {model} ({storage}) {color}",
    "{brand} {model} {color} {storage}",
    "{model} {storage} {color} - {brand}",
    "{brand} {model} {storage}, {color}",
]


def _fmt_storage(storage: str, alt: bool) -> str:
    if storage is None:
        return ""
    if "GB" in storage and "SSD" not in storage:
        num = storage.replace("GB", "")
        return f"{num} GB" if alt else f"{num}GB"
    return storage


def _make_title(brand, model, storage, color, extra, alt: bool) -> str:
    storage_str = _fmt_storage(storage, alt)

    # Build from parts and only use templates with parentheses/commas
    # when there's actually a storage value to wrap -- otherwise fall
    # back to a plain space-joined template to avoid "()" artifacts.
    if storage_str:
        template = random.choice(TITLE_TEMPLATES)
        parts = template.format(brand=brand, model=model, storage=storage_str, color=color)
    else:
        plain_templates = [
            "{brand} {model} {color}",
            "{brand} {model}, {color}",
            "{model} {color} - {brand}",
        ]
        template = random.choice(plain_templates)
        parts = template.format(brand=brand, model=model, color=color)

    parts = " ".join(parts.split())
    if extra:
        parts = f"{parts} {extra}" if random.random() < 0.5 else f"{extra} {parts}"
    return parts


SPEC_ATTRS = {
    "smartphone": ["chip", "display", "camera"],
    "laptop": ["cpu_gpu", "ram"],
    "earbuds": ["battery", "feature"],
    "smartwatch": ["battery", "feature"],
    "television": ["resolution", "size"],
    "footwear": ["material", "size"],
    "kitchen_appliance": ["power", "capacity"],
    "gaming_console": ["storage_perf", "graphics"],
    "camera": ["sensor", "resolution_mp"],
    "monitor": ["refresh", "resolution"],
    "storage_device": ["speed", "interface"],
    "vacuum": ["suction", "runtime"],
}

SPEC_FILLERS = {
    "chip": ["A16 Bionic chip", "Snapdragon 8 Gen 3", "Dimensity 9200", "Exynos 2400", "Tensor G3"],
    "display": ["6.1-inch OLED display", "6.7 inch AMOLED", "6.5in LCD 120Hz", "6.1 Super Retina XDR"],
    "camera": ["48MP main camera", "50 MP triple camera", "108MP quad camera"],
    "cpu_gpu": ["Ryzen 7 RTX 4060", "Core i5 12th Gen", "Ryzen 5 5600H", "Core i7 RTX 3050"],
    "ram": ["16GB RAM", "8GB RAM", "32 GB RAM"],
    "battery": ["Up to 30 hours battery", "22hr playback", "ANC enabled"],
    "feature": ["Bluetooth 5.3", "IP67 water resistant", "Heart rate monitor"],
    "resolution": ["4K UHD resolution", "Full HD 1080p"],
    "size": ["43 inch screen", "55in display"],
    "material": ["Mesh upper", "Leather upper", "breathable knit"],
    "power": ["750W motor", "1400 Watt"],
    "capacity": ["1.5L capacity", "2 Litre"],
    "storage_perf": ["Custom SSD storage", "Fast load times"],
    "graphics": ["4K graphics support", "Ray tracing support", "120fps gaming"],
    "sensor": ["APS-C sensor", "Full-frame sensor", "1-inch sensor"],
    "resolution_mp": ["24.2MP resolution", "33MP resolution", "26MP resolution"],
    "refresh": ["144Hz refresh rate", "165Hz refresh rate", "60Hz refresh rate"],
    "resolution": ["4K UHD resolution", "QHD resolution", "Full HD resolution"],
    "speed": ["7000 MB/s read speed", "560 MB/s read speed", "1050 MB/s transfer speed"],
    "interface": ["PCIe 4.0 interface", "USB-C interface", "SATA III interface"],
    "suction": ["230AW suction power", "Powerful cyclone suction"],
    "runtime": ["60 minute runtime", "40 minute runtime", "90 minute runtime"],
}


def _pick_spec_values(category: str) -> dict:
    """Chooses concrete spec attribute values for one product instance."""
    attrs = SPEC_ATTRS.get(category, [])
    chosen_attrs = random.sample(attrs, k=min(2, len(attrs))) if attrs else []
    return {a: random.choice(SPEC_FILLERS[a]) for a in chosen_attrs}


def _format_specs(values: dict, storage, extra, alt: bool) -> str:
    """Renders a chosen set of spec values into a phrased string. Two
    calls with the SAME `values` dict produce two different-looking but
    factually-consistent spec strings (for positive pairs); two calls
    with DIFFERENT `values` dicts produce factually different specs
    (for negative pairs)."""
    pieces = list(values.values())
    if storage:
        pieces.append(_fmt_storage(storage, alt))
    if extra and random.random() < 0.5:
        pieces.append(extra)
    pieces = pieces[:]  # copy before shuffling
    random.shuffle(pieces)
    sep = ", " if alt else " | "
    return sep.join(pieces)


def generate_positive_pair(category: str):
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    storage = random.choice(cat["storages"])
    color_a, color_b = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model, storage, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage, color_b, extra, alt=True)
    return title_a, title_b, 1


def generate_hard_negative_pair(category: str):
    """Same brand + model family, but a different variant -> different product."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    storage_a = random.choice(cat["storages"])
    storage_b = random.choice(cat["storages"])
    color_a, _ = random.choice(cat["colors"])
    color_b, _ = random.choice(cat["colors"])

    # Force at least one real difference (storage or color) so it's not
    # accidentally identical.
    tries = 0
    while storage_a == storage_b and color_a == color_b and tries < 5:
        storage_b = random.choice(cat["storages"])
        color_b = random.choice(cat["colors"])[0]
        tries += 1

    title_a = _make_title(brand, model, storage_a, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage_b, color_b, extra, alt=False)
    return title_a, title_b, 0


def generate_easy_negative_pair(category_a: str, category_b: str):
    a_brand, a_models = random.choice(CATEGORIES[category_a]["brands_models"])
    a_model = random.choice(a_models)
    a_storage = random.choice(CATEGORIES[category_a]["storages"])
    a_color, _ = random.choice(CATEGORIES[category_a]["colors"])
    a_extra = random.choice(CATEGORIES[category_a].get("extra", [None])) if "extra" in CATEGORIES[category_a] else None
    title_a = _make_title(a_brand, a_model, a_storage, a_color, a_extra, alt=False)

    b_brand, b_models = random.choice(CATEGORIES[category_b]["brands_models"])
    b_model = random.choice(b_models)
    b_storage = random.choice(CATEGORIES[category_b]["storages"])
    b_color, _ = random.choice(CATEGORIES[category_b]["colors"])
    b_extra = random.choice(CATEGORIES[category_b].get("extra", [None])) if "extra" in CATEGORIES[category_b] else None
    title_b = _make_title(b_brand, b_model, b_storage, b_color, b_extra, alt=False)

    return title_a, title_b, 0


def generate_dataset(n_total: int):
    categories = list(CATEGORIES.keys())
    rows = []
    seen = set()

    # Target mix: 40% positive, 35% hard negative, 25% easy negative
    n_pos = int(n_total * 0.40)
    n_hard_neg = int(n_total * 0.35)
    n_easy_neg = n_total - n_pos - n_hard_neg

    def _add(gen_fn, count):
        attempts, added = 0, 0
        while added < count and attempts < count * 20:
            attempts += 1
            row = gen_fn()
            key = (row[0].lower(), row[1].lower())
            key_rev = (row[1].lower(), row[0].lower())
            if key in seen or key_rev in seen or row[0].lower() == row[1].lower():
                continue
            seen.add(key)
            rows.append(row)
            added += 1

    _add(lambda: generate_positive_pair(random.choice(categories)), n_pos)
    _add(lambda: generate_hard_negative_pair(random.choice(categories)), n_hard_neg)
    _add(lambda: generate_easy_negative_pair(*random.sample(categories, 2)), n_easy_neg)

    random.shuffle(rows)
    return rows


DESC_TEMPLATES = [
    "The {brand} {model} comes with {spec_text}.",
    "{brand}'s {model} features {spec_text} for everyday use.",
    "Experience {spec_text} with the {brand} {model}.",
    "This {model} from {brand} is equipped with {spec_text}.",
    "{model} by {brand}: {spec_text}.",
]


def _make_description(brand, model, spec_values: dict, storage, extra, alt: bool) -> str:
    pieces = list(spec_values.values())
    if storage:
        pieces.append(_fmt_storage(storage, alt))
    if extra:
        pieces.append(extra)
    pieces = pieces[:]
    random.shuffle(pieces)
    spec_text = ", ".join(pieces) if pieces else "premium build quality"
    template = random.choice(DESC_TEMPLATES)
    return template.format(brand=brand, model=model, spec_text=spec_text)


_ID_COUNTER = {"n": 0}


def _next_id() -> str:
    _ID_COUNTER["n"] += 1
    return f"P{_ID_COUNTER['n']:06d}"


def generate_positive_pair_structured(category: str):
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

    # Same brand on both sides (real field now, not just embedded in title).
    return (_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 1


def generate_hard_negative_pair_structured(category: str):
    """Same brand, different variant -> different product, different specs."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model_a = random.choice(models)
    model_b = random.choice(models)
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    storage_a = random.choice(cat["storages"])
    storage_b = random.choice(cat["storages"])
    color_a, _ = random.choice(cat["colors"])
    color_b, _ = random.choice(cat["colors"])

    tries = 0
    while model_a == model_b and storage_a == storage_b and color_a == color_b and tries < 5:
        model_b = random.choice(models)
        storage_b = random.choice(cat["storages"])
        tries += 1

    title_a = _make_title(brand, model_a, storage_a, color_a, extra, alt=False)
    title_b = _make_title(brand, model_b, storage_b, color_b, extra, alt=False)
    desc_a = _make_description(brand, model_a, _pick_spec_values(category), storage_a, extra, alt=False)
    desc_b = _make_description(brand, model_b, _pick_spec_values(category), storage_b, extra, alt=False)

    # Same brand (both sides), different underlying product -> tests that
    # the model doesn't shortcut on "brand matches => same product".
    return (_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0


def generate_easy_negative_pair_structured(category_a: str, category_b: str):
    a_brand, a_models = random.choice(CATEGORIES[category_a]["brands_models"])
    a_model = random.choice(a_models)
    a_storage = random.choice(CATEGORIES[category_a]["storages"])
    a_color, _ = random.choice(CATEGORIES[category_a]["colors"])
    a_extra = random.choice(CATEGORIES[category_a].get("extra", [None])) if "extra" in CATEGORIES[category_a] else None
    title_a = _make_title(a_brand, a_model, a_storage, a_color, a_extra, alt=False)
    desc_a = _make_description(a_brand, a_model, _pick_spec_values(category_a), a_storage, a_extra, alt=False)

    b_brand, b_models = random.choice(CATEGORIES[category_b]["brands_models"])
    b_model = random.choice(b_models)
    b_storage = random.choice(CATEGORIES[category_b]["storages"])
    b_color, _ = random.choice(CATEGORIES[category_b]["colors"])
    b_extra = random.choice(CATEGORIES[category_b].get("extra", [None])) if "extra" in CATEGORIES[category_b] else None
    title_b = _make_title(b_brand, b_model, b_storage, b_color, b_extra, alt=False)
    desc_b = _make_description(b_brand, b_model, _pick_spec_values(category_b), b_storage, b_extra, alt=False)

    return (_next_id(), title_a, a_brand, desc_a), (_next_id(), title_b, b_brand, desc_b), 0


def generate_dataset_structured(n_total: int):
    categories = list(CATEGORIES.keys())
    rows = []
    seen = set()

    n_pos = int(n_total * 0.40)
    n_hard_neg = int(n_total * 0.35)
    n_easy_neg = n_total - n_pos - n_hard_neg

    def _add(gen_fn, count):
        attempts, added = 0, 0
        while added < count and attempts < count * 20:
            attempts += 1
            side_a, side_b, label = gen_fn()
            title_a, title_b = side_a[1], side_b[1]
            key, key_rev = (title_a.lower(), title_b.lower()), (title_b.lower(), title_a.lower())
            if key in seen or key_rev in seen or title_a.lower() == title_b.lower():
                continue
            seen.add(key)
            rows.append((*side_a, *side_b, label))
            added += 1

    _add(lambda: generate_positive_pair_structured(random.choice(categories)), n_pos)
    _add(lambda: generate_hard_negative_pair_structured(random.choice(categories)), n_hard_neg)
    _add(lambda: generate_easy_negative_pair_structured(*random.sample(categories, 2)), n_easy_neg)

    random.shuffle(rows)
    return rows



    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    storage = random.choice(cat["storages"])
    color_a, color_b = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model, storage, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage, color_b, extra, alt=True)
    spec_values = _pick_spec_values(category)
    specs_a = _format_specs(spec_values, storage, extra, alt=False)
    specs_b = _format_specs(spec_values, storage, extra, alt=True)
    return title_a, specs_a, title_b, specs_b, 1


def generate_hard_negative_pair_with_specs(category: str):
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    storage_a = random.choice(cat["storages"])
    storage_b = random.choice(cat["storages"])
    color_a, _ = random.choice(cat["colors"])
    color_b, _ = random.choice(cat["colors"])

    tries = 0
    while storage_a == storage_b and color_a == color_b and tries < 5:
        storage_b = random.choice(cat["storages"])
        color_b = random.choice(cat["colors"])[0]
        tries += 1

    title_a = _make_title(brand, model, storage_a, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage_b, color_b, extra, alt=False)
    specs_a = _format_specs(_pick_spec_values(category), storage_a, extra, alt=False)
    specs_b = _format_specs(_pick_spec_values(category), storage_b, extra, alt=False)
    return title_a, specs_a, title_b, specs_b, 0


def generate_easy_negative_pair_with_specs(category_a: str, category_b: str):
    a_brand, a_models = random.choice(CATEGORIES[category_a]["brands_models"])
    a_model = random.choice(a_models)
    a_storage = random.choice(CATEGORIES[category_a]["storages"])
    a_color, _ = random.choice(CATEGORIES[category_a]["colors"])
    a_extra = random.choice(CATEGORIES[category_a].get("extra", [None])) if "extra" in CATEGORIES[category_a] else None
    title_a = _make_title(a_brand, a_model, a_storage, a_color, a_extra, alt=False)
    specs_a = _format_specs(_pick_spec_values(category_a), a_storage, a_extra, alt=False)

    b_brand, b_models = random.choice(CATEGORIES[category_b]["brands_models"])
    b_model = random.choice(b_models)
    b_storage = random.choice(CATEGORIES[category_b]["storages"])
    b_color, _ = random.choice(CATEGORIES[category_b]["colors"])
    b_extra = random.choice(CATEGORIES[category_b].get("extra", [None])) if "extra" in CATEGORIES[category_b] else None
    title_b = _make_title(b_brand, b_model, b_storage, b_color, b_extra, alt=False)
    specs_b = _format_specs(_pick_spec_values(category_b), b_storage, b_extra, alt=False)

    return title_a, specs_a, title_b, specs_b, 0


def generate_dataset_with_specs(n_total: int):
    categories = list(CATEGORIES.keys())
    rows = []
    seen = set()

    n_pos = int(n_total * 0.40)
    n_hard_neg = int(n_total * 0.35)
    n_easy_neg = n_total - n_pos - n_hard_neg

    def _add(gen_fn, count):
        attempts, added = 0, 0
        while added < count and attempts < count * 20:
            attempts += 1
            row = gen_fn()
            key = (row[0].lower(), row[2].lower())
            key_rev = (row[2].lower(), row[0].lower())
            if key in seen or key_rev in seen or row[0].lower() == row[2].lower():
                continue
            seen.add(key)
            rows.append(row)
            added += 1

    _add(lambda: generate_positive_pair_with_specs(random.choice(categories)), n_pos)
    _add(lambda: generate_hard_negative_pair_with_specs(random.choice(categories)), n_hard_neg)
    _add(lambda: generate_easy_negative_pair_with_specs(*random.sample(categories, 2)), n_easy_neg)

    random.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200, help="Total number of rows to generate")
    parser.add_argument("--out", type=str, default="data/products_1000.csv")
    parser.add_argument(
        "--schema", choices=["title_only", "with_specs", "structured"], default="title_only",
        help="title_only -> product1,product2,label. "
             "with_specs -> product1_title,product1_specs,product2_title,product2_specs,label. "
             "structured -> product1_id,product1_title,product1_brand,product1_description,"
             "product2_id,product2_title,product2_brand,product2_description,label "
             "(recommended: brand is checked as its own explicit signal, not just embedded in the title)."
    )
    args = parser.parse_args()

    if args.schema == "structured":
        rows = generate_dataset_structured(args.n)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "product1_id", "product1_title", "product1_brand", "product1_description",
                "product2_id", "product2_title", "product2_brand", "product2_description",
                "label",
            ])
            writer.writerows(rows)
        n_pos = sum(1 for r in rows if r[-1] == 1)
    elif args.schema == "with_specs":
        rows = generate_dataset_with_specs(args.n)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product1_title", "product1_specs", "product2_title", "product2_specs", "label"])
            writer.writerows(rows)
        n_pos = sum(1 for r in rows if r[4] == 1)
    else:
        rows = generate_dataset(args.n)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product1", "product2", "label"])
            writer.writerows(rows)
        n_pos = sum(1 for r in rows if r[2] == 1)

    n_neg = len(rows) - n_pos
    print(f"Wrote {len(rows)} rows to {args.out} (schema={args.schema}, positives={n_pos}, negatives={n_neg})")


if __name__ == "__main__":
    main()