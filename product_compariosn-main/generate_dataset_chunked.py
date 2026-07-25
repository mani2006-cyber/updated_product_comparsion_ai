"""
generate_dataset_chunked.py
===========================
Generates a high-quality product-matching dataset in CHUNKS.
Each chunk is saved as a separate CSV so you never hit memory/session limits.

Run locally in VS Code:
    python generate_dataset_chunked.py --total 100000 --chunk-size 5000 --out-dir ./data_chunks

This creates:
    ./data_chunks/products_chunk_000.csv  (5,000 rows)
    ./data_chunks/products_chunk_001.csv  (5,000 rows)
    ...
    ./data_chunks/products_chunk_019.csv  (5,000 rows)

Then merge:
    python merge_chunks.py --input-dir ./data_chunks --output data/products_combined.csv
"""

import argparse
import csv
import itertools
import os
import random
from typing import List, Tuple

random.seed(42)


# ==========================================================================
# CATEGORY DEFINITIONS
# ==========================================================================
CATEGORIES = {
    "smartphone": {
        "brands_models": [
            ("Apple", ["iPhone 13", "iPhone 14", "iPhone 15", "iPhone 15 Pro", "iPhone 16", "iPhone 16 Pro", "iPhone SE 3rd Gen"]),
            ("Samsung", ["Galaxy S23", "Galaxy S23 Ultra", "Galaxy S24", "Galaxy S24 Ultra", "Galaxy A54", "Galaxy Z Flip5", "Galaxy Z Fold5"]),
            ("OnePlus", ["11", "12", "Nord 3", "12R", "Open"]),
            ("Xiaomi", ["Redmi Note 13 Pro", "Redmi Note 12", "Mi 13", "Poco X6 Pro", "Xiaomi 14", "Xiaomi 14 Ultra"]),
            ("Google", ["Pixel 7", "Pixel 8", "Pixel 8 Pro", "Pixel 8a", "Pixel 9"]),
            ("Vivo", ["V29", "V27 Pro", "X90", "X100"]),
            ("Oppo", ["Reno 10", "Reno 11 Pro", "F25 Pro", "Find X7"]),
            ("Realme", ["11 Pro", "GT 5", "Narzo 60", "GT 6"]),
            ("Motorola", ["Edge 40", "Edge 50 Pro", "Moto G84", "Razr 40 Ultra"]),
            ("Nothing", ["Phone 1", "Phone 2", "Phone 2a"]),
            ("Sony", ["Xperia 1 V", "Xperia 5 V", "Xperia 10 V"]),
            ("Asus", ["ROG Phone 8", "Zenfone 10"]),
        ],
        "storages": ["64GB", "128GB", "256GB", "512GB", "1TB"],
        "colors": [
            ("Black", "Black"), ("Titanium Gray", "Titanium Grey"), ("Blue", "Blue"),
            ("Green", "Green"), ("White", "White"), ("Midnight", "Midnight"),
            ("Desert Titanium", "Desert Titanium"), ("Phantom Black", "Phantom Black"),
            ("Silver", "Silver"), ("Gold", "Gold"), ("Purple", "Purple"),
        ],
        "extra": ["New", "Renewed", "Refurbished", "Open Box", None],
    },
    "laptop": {
        "brands_models": [
            ("HP", ["Victus", "Pavilion 15", "Envy x360", "Omen 16", "Spectre x360"]),
            ("Dell", ["Inspiron 15", "XPS 13", "Vostro 14", "Alienware m16", "Latitude 7430"]),
            ("Lenovo", ["ThinkPad E14", "IdeaPad Slim 5", "Legion 5", "Yoga Slim 7", "ThinkPad X1 Carbon"]),
            ("Asus", ["Vivobook 15", "ROG Strix G16", "Zenbook 14", "TUF Gaming A15", "ProArt P16"]),
            ("Acer", ["Aspire 7", "Nitro 5", "Swift 3", "Predator Helios", "Aspire 5"]),
            ("Apple", ["MacBook Air M2", "MacBook Air M3", "MacBook Pro 14 M3", "MacBook Pro 16 M3 Max"]),
            ("MSI", ["Katana 15", "Modern 14", "Cyborg 15", "Stealth 16"]),
            ("Razer", ["Blade 14", "Blade 16", "Blade 18"]),
        ],
        "storages": ["256GB SSD", "512GB SSD", "1TB SSD", "2TB SSD"],
        "colors": [("Silver", "Silver"), ("Black", "Black"), ("Space Gray", "Space Grey"), ("Blue", "Blue"), ("White", "White")],
        "extra": ["Ryzen 7 RTX 4060", "Core i5 RTX 3050", "Ryzen 5", "Core i7 16GB RAM", "Core i9 RTX 4070",
                   "Windows 11", "Ubuntu Linux", "Touch Display", "Non-Touch Display", "OLED Display"],
    },
    "earbuds": {
        "brands_models": [
            ("Boat", ["Airdopes 311 Pro", "Airdopes 141", "Airdopes 441", "Rockerz 255", "Airdopes 300"]),
            ("JBL", ["Tune 230NC", "Wave Buds", "Tour Pro 2", "Live Pro 2", "Tune 760NC"]),
            ("Sony", ["WF-1000XM4", "WF-1000XM5", "WF-C700N", "WF-C500"]),
            ("Apple", ["AirPods Pro 2", "AirPods 3rd Gen", "AirPods 4", "AirPods Max"]),
            ("Samsung", ["Galaxy Buds2 Pro", "Galaxy Buds FE", "Galaxy Buds3"]),
            ("Noise", ["Buds VS104", "Air Buds", "Buds Prima", "Buds X"]),
            ("Realme", ["Buds Air 5", "Buds T300", "Buds Q3s"]),
            ("pTron", ["Bassbuds Astra", "Bassbuds Duo", "Bassbuds Sports", "Bassbuds Pixel"]),
            ("Zebronics", ["Sound Bomb 1", "Sound Bomb Q Pro", "Sound Bomb 7", "Sound Bomb 4"]),
            ("GOBOULT", ["Z40 Pro", "AirBass K60", "Z40"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("White", "White"), ("Blue", "Blue"), ("Teal", "Teal"), ("Pink", "Pink")],
    },
    "smartwatch": {
        "brands_models": [
            ("Apple", ["Watch Series 9", "Watch SE", "Watch Ultra 2", "Watch Series 10"]),
            ("Samsung", ["Galaxy Watch6", "Galaxy Watch6 Classic", "Galaxy Watch FE", "Galaxy Watch Ultra"]),
            ("Noise", ["ColorFit Pulse 2", "ColorFit Ultra 3", "ColorFit Icon 2", "Fit Active"]),
            ("Boat", ["Wave Neo", "Xtend", "Storm Pro", "Lunar Pro"]),
            ("Fire-Boltt", ["Phoenix Pro", "Ninja Call Pro", "Talk 2", "Visionary"]),
            ("Fitbit", ["Versa 4", "Sense 2", "Charge 6", "Inspire 3"]),
            ("Garmin", ["Forerunner 265", "Venu 3", "Fenix 7"]),
            ("Amazfit", ["GTR 4", "Bip 3 Pro", "GTS 4 Mini"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("Rose Gold", "Rose Gold"), ("Blue", "Blue"), ("Green", "Green")],
    },
    "television": {
        "brands_models": [
            ("Samsung", ["Crystal 4K UHD", "The Frame QLED", "Neo QLED 4K", "OLED S95D"]),
            ("LG", ["UQ7500 4K", "OLED C3", "NanoCell 4K", "OLED C4"]),
            ("Sony", ["Bravia X75K", "Bravia X90L", "Bravia A80L", "Bravia 7"]),
            ("Mi", ["X Pro 4K", "5A Pro", "OLED Vision"]),
            ("OnePlus", ["Y1S Pro", "U1S", "Q2 Pro"]),
            ("TCL", ["C645 QLED", "C755 Mini LED", "P635"]),
            ("Hisense", ["U7K", "A7K", "E7K"]),
        ],
        "storages": ["43 inch", "50 inch", "55 inch", "65 inch", "75 inch", "85 inch"],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("Titan Gray", "Titan Grey")],
    },
    "footwear": {
        "brands_models": [
            ("Nike", ["Air Max 270", "Revolution 6", "Air Force 1", "Pegasus 40", "Dunk Low", "Jordan 1"]),
            ("Adidas", ["Ultraboost 22", "Duramo SL", "Superstar", "Runfalcon 3", "Samba OG"]),
            ("Puma", ["Softride Rift", "Smash v2", "Anzarun Lite", "RS-X"]),
            ("Reebok", ["Classic Leather", "Energen Lux", "Flexagon Force", "Club C"]),
            ("Skechers", ["Go Walk 6", "Summits", "Flex Advantage", "Max Cushioning"]),
            ("New Balance", ["574", "327", "Fresh Foam X 1080", "550"]),
            ("Asics", ["Gel-Kayano 30", "Gel-Nimbus 26", "GT-2000 12"]),
        ],
        "storages": ["UK 6", "UK 7", "UK 8", "UK 9", "UK 10", "UK 11", "UK 12"],
        "colors": [
            ("Black/White", "Black/White"), ("Triple Black", "Triple Black"),
            ("Grey", "Gray"), ("Blue", "Blue"), ("Red", "Red"), ("Green", "Green"),
        ],
    },
    "kitchen_appliance": {
        "brands_models": [
            ("Prestige", ["Mixer Grinder 750W", "Induction Cooktop", "Electric Kettle 1.5L", "Pressure Cooker 5L"]),
            ("Philips", ["Air Fryer HD9252", "Mixer Grinder HL7756", "Juicer HR1832", "Hand Blender"]),
            ("Bajaj", ["Majesty Toaster", "Mixer Grinder Rex", "Induction Cooktop 1400W", "Electric Rice Cooker"]),
            ("Havells", ["Toaster ST-11", "Electric Kettle Aquis", "Mixer Grinder"]),
            ("Butterfly", ["Mixer Grinder Rocket", "Induction Cooktop Rapid", "Wet Grinder"]),
            ("Bosch", ["Mixer Grinder", "Dishwasher SMS4HTI31I"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("Red", "Red"), ("White", "White"), ("Steel", "Steel"), ("Silver", "Silver")],
    },
    "gaming_console": {
        "brands_models": [
            ("Sony", ["PlayStation 5", "PlayStation 5 Pro", "PlayStation 4", "PlayStation 5 Slim"]),
            ("Microsoft", ["Xbox Series X", "Xbox Series S", "Xbox Series X 1TB"]),
            ("Nintendo", ["Switch OLED", "Switch Lite", "Switch 2", "Switch"]),
            ("Steam", ["Steam Deck OLED", "Steam Deck LCD"]),
            ("ASUS", ["ROG Ally", "ROG Ally X"]),
        ],
        "storages": ["512GB Disc Edition", "512GB Digital Edition", "1TB", "2TB"],
        "colors": [("White", "White"), ("Black", "Black")],
    },
    "camera": {
        "brands_models": [
            ("Canon", ["EOS R50", "EOS R10", "EOS R6 Mark II", "EOS 90D", "EOS R5"]),
            ("Sony", ["Alpha a7 IV", "Alpha a6400", "ZV-E10", "Alpha a7C II", "FX30"]),
            ("Nikon", ["Z50", "Z6 III", "D7500", "Zf", "Z8"]),
            ("Fujifilm", ["X-T5", "X-S20", "X100VI", "X-H2"]),
            ("Panasonic", ["Lumix GH6", "Lumix S5 II"]),
        ],
        "storages": ["Body Only", "with 18-55mm Lens", "with 24-70mm Lens", "with 50mm Lens"],
        "colors": [("Black", "Black"), ("Silver", "Silver")],
        "extra": ["Mirrorless Camera", "DSLR Camera", "APS-C Sensor", "Full Frame Sensor", "4K Video"],
    },
    "monitor": {
        "brands_models": [
            ("Dell", ["UltraSharp U2723QE", "UltraSharp U3223QE", "S2721DGF", "Alienware AW3423DWF"]),
            ("LG", ["27GP850", "34WP65C", "UltraGear 27GN800", "27UP850N"]),
            ("Samsung", ["Odyssey G7", "ViewFinity S8", "Odyssey Neo G9", "Smart Monitor M8"]),
            ("ASUS", ["ProArt PA278QV", "TUF Gaming VG27AQ", "ROG Swift PG27AQN"]),
            ("BenQ", ["EW2880U", "PD2705U", "EX2710Q"]),
        ],
        "storages": ["24 inch", "27 inch", "32 inch", "34 inch", "49 inch"],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("White", "White")],
    },
    "storage_device": {
        "brands_models": [
            ("Samsung", ["990 PRO", "970 EVO Plus", "T7 Shield", "T9 Portable"]),
            ("WD", ["Black SN850X", "Blue SN580", "My Passport", "Elements SE"]),
            ("Crucial", ["P5 Plus", "MX500", "X6 Portable", "T500"]),
            ("SanDisk", ["Extreme Portable", "Extreme Pro", "Ultra Dual Drive"]),
            ("Seagate", ["FireCuda 530", "Barracuda", "Expansion Card"]),
        ],
        "storages": ["500GB", "1TB", "2TB", "4TB", "8TB"],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("Blue", "Blue")],
        "extra": ["NVMe SSD", "PCIe 4.0", "Portable SSD", "SATA SSD", "External HDD"],
    },
    "vacuum": {
        "brands_models": [
            ("Dyson", ["V15 Detect", "V12 Detect Slim", "V8", "Ball Animal 3", "Gen5detect"]),
            ("Shark", ["Navigator Lift-Away", "Vertex Pro", "Stratos", "Rocket Pet Pro"]),
            ("iRobot", ["Roomba j7+", "Roomba 694", "Roomba Combo j9+", "Roomba s9+"]),
            ("Eureka", ["NEC122", "PowerSpeed", "WhirlWind"]),
            ("Xiaomi", ["Dreame L10s Ultra", "Roborock S8 Pro Ultra"]),
        ],
        "storages": [None],
        "colors": [("Yellow/Nickel", "Yellow/Nickel"), ("Black", "Black"), ("Silver", "Silver"), ("Blue", "Blue")],
        "extra": ["Cordless Vacuum", "Robot Vacuum", "Bagless Upright",
                   "with crevice tool", "with pet hair tool", "with extra filter"],
    },
    "book": {
        "brands_models": [
            ("George Orwell", ["1984", "Animal Farm", "Homage to Catalonia"]),
            ("J.R.R. Tolkien", ["The Hobbit", "The Fellowship of the Ring", "The Two Towers", "The Return of the King"]),
            ("Jane Austen", ["Pride and Prejudice", "Sense and Sensibility", "Emma"]),
            ("Suzanne Collins", ["The Hunger Games", "Catching Fire", "Mockingjay"]),
            ("J.K. Rowling", ["Harry Potter and the Sorcerer's Stone", "Harry Potter and the Chamber of Secrets"]),
            ("Stephen King", ["The Shining", "It", "The Stand"]),
        ],
        "storages": [None],
        "colors": [("Paperback", "Paperback"), ("Hardcover", "Hardcover"), ("Kindle", "Kindle")],
        "extra": ["1st Edition", "2nd Edition", "3rd Edition", "Anniversary Edition",
                   "Illustrated Edition", "Collector's Edition", None],
    },
    "video_game": {
        "brands_models": [
            ("EA Sports", ["FIFA 23", "Madden NFL 24", "FC 24", "WWE 2K24"]),
            ("CD Projekt Red", ["The Witcher 3", "Cyberpunk 2077", "GWENT"]),
            ("Rockstar Games", ["Grand Theft Auto V", "Red Dead Redemption 2", "GTA 6"]),
            ("Nintendo", ["The Legend of Zelda: Tears of the Kingdom", "Super Mario Odyssey", "Mario Kart 8 Deluxe"]),
            ("FromSoftware", ["Elden Ring", "Dark Souls III", "Sekiro"]),
            ("Bethesda", ["Starfield", "The Elder Scrolls V: Skyrim", "Fallout 4"]),
        ],
        "storages": [None],
        "colors": [("Standard", "Standard"), ("Deluxe", "Deluxe")],
        "extra": ["PS5", "Xbox Series X", "Nintendo Switch", "PC", "PS4",
                   "Standard Edition", "Game of the Year Edition", "Deluxe Edition", "Ultimate Edition"],
    },
    "networking": {
        "brands_models": [
            ("TP-Link", ["Archer AX50", "Archer AX21", "Deco X60", "Archer AX73"]),
            ("Netgear", ["Nighthawk AX12", "Orbi RBK852", "Nighthawk RAXE500"]),
            ("ASUS", ["RT-AX88U", "ZenWiFi AX", "ROG Rapture GT-AXE16000"]),
            ("Linksys", ["Hydra Pro 6E", "Atlas Max 6E"]),
            ("Ubiquiti", ["UniFi Dream Machine", "UniFi 6 Pro"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("White", "White")],
        "extra": ["Wi-Fi 6", "Wi-Fi 6E", "Wi-Fi 7", "Dual-Band", "Tri-Band", "v1", "v2", "v3"],
    },
    "headphones": {
        "brands_models": [
            ("Sony", ["WH-1000XM5", "WH-1000XM4", "WH-CH720N", "MDR-7506"]),
            ("Bose", ["QuietComfort 45", "QuietComfort Ultra", "SoundLink Around-Ear"]),
            ("Sennheiser", ["Momentum 4", "HD 560S", "HD 660S2"]),
            ("Audio-Technica", ["ATH-M50x", "ATH-R70x", "ATH-MSR7"]),
            ("JBL", ["Tune 760NC", "Live 660NC", "Tour One M2"]),
            ("Skullcandy", ["Crusher ANC 2", "Hesh ANC"]),
        ],
        "storages": [None],
        "colors": [("Black", "Black"), ("Silver", "Silver"), ("Blue", "Blue"), ("White", "White")],
        "extra": ["Active Noise Cancellation", "Transparency Mode", "Bluetooth 5.2", "Hi-Res Audio"],
    },
    "tablet": {
        "brands_models": [
            ("Apple", ["iPad Air M2", "iPad Pro 11 M4", "iPad Pro 13 M4", "iPad Mini 6"]),
            ("Samsung", ["Galaxy Tab S9", "Galaxy Tab S9 Ultra", "Galaxy Tab A9+"]),
            ("Xiaomi", ["Pad 6", "Pad 6 Pro", "Redmi Pad SE"]),
            ("Lenovo", ["Tab P12", "Tab M10 Plus", "Yoga Tab 13"]),
            ("Microsoft", ["Surface Pro 9", "Surface Go 3", "Surface Pro 10"]),
        ],
        "storages": ["64GB", "128GB", "256GB", "512GB", "1TB"],
        "colors": [("Space Gray", "Space Grey"), ("Silver", "Silver"), ("Blue", "Blue"), ("Pink", "Pink")],
        "extra": ["Wi-Fi", "Wi-Fi + Cellular", "with Keyboard", "with Pen"],
    },
}


TITLE_TEMPLATES = [
    "{brand} {model} {storage} {color}",
    "{brand} {model} ({storage}) {color}",
    "{brand} {model} {color} {storage}",
    "{model} {storage} {color} - {brand}",
    "{brand} {model} {storage}, {color}",
    "{color} {brand} {model} {storage}",
    "{brand} {model} - {color} ({storage})",
]

DESC_TEMPLATES = [
    "The {brand} {model} comes with {spec_text}.",
    "{brand}'s {model} features {spec_text} for everyday use.",
    "Experience {spec_text} with the {brand} {model}.",
    "This {model} from {brand} is equipped with {spec_text}.",
    "{model} by {brand}: {spec_text}.",
    "Discover the {brand} {model} with {spec_text}.",
]

SPEC_FILLERS = {
    "chip": ["A16 Bionic chip", "Snapdragon 8 Gen 3", "Dimensity 9200", "Exynos 2400", "Tensor G3", "Apple M3 chip"],
    "display": ["6.1-inch OLED display", "6.7 inch AMOLED", "6.5in LCD 120Hz", "6.1 Super Retina XDR", "LTPO 120Hz"],
    "camera": ["48MP main camera", "50 MP triple camera", "108MP quad camera", "200MP main sensor"],
    "cpu_gpu": ["Ryzen 7 RTX 4060", "Core i5 12th Gen", "Ryzen 5 5600H", "Core i7 RTX 3050", "Apple M3 Pro GPU"],
    "ram": ["16GB RAM", "8GB RAM", "32 GB RAM", "64GB RAM"],
    "battery": ["Up to 30 hours battery", "22hr playback", "ANC enabled", "60 hour total playtime"],
    "feature": ["Bluetooth 5.3", "IP67 water resistant", "Heart rate monitor", "SpO2 sensor", "GPS tracking"],
    "resolution": ["4K UHD resolution", "Full HD 1080p", "2K QHD", "8K UHD"],
    "size": ["43 inch screen", "55in display", "65 inch panel", "75 inch screen"],
    "material": ["Mesh upper", "Leather upper", "breathable knit", "synthetic leather"],
    "power": ["750W motor", "1400 Watt", "2000W", "1200W"],
    "capacity": ["1.5L capacity", "2 Litre", "5L", "3.5L"],
    "storage_perf": ["Custom SSD storage", "Fast load times", "NVMe Gen4"],
    "graphics": ["4K graphics support", "Ray tracing support", "120fps gaming", "8K gaming"],
    "sensor": ["APS-C sensor", "Full-frame sensor", "1-inch sensor", "Micro Four Thirds"],
    "resolution_mp": ["24.2MP resolution", "33MP resolution", "26MP resolution", "45.7MP"],
    "refresh": ["144Hz refresh rate", "165Hz refresh rate", "60Hz refresh rate", "240Hz"],
    "speed": ["7000 MB/s read speed", "560 MB/s read speed", "1050 MB/s transfer speed", "2000MB/s"],
    "interface": ["PCIe 4.0 interface", "USB-C interface", "SATA III interface", "Thunderbolt 4"],
    "suction": ["230AW suction power", "Powerful cyclone suction", "25KPa suction"],
    "runtime": ["60 minute runtime", "40 minute runtime", "90 minute runtime", "120 minute"],
}

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
    "book": ["extra"],
    "video_game": ["extra"],
    "networking": ["extra"],
    "headphones": ["battery", "feature"],
    "tablet": ["chip", "display"],
}


# ==========================================================================
# HELPERS
# ==========================================================================
_id_counter = 0

def _next_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"P{_id_counter:08d}"


def _fmt_storage(storage: str, alt: bool) -> str:
    if storage is None:
        return ""
    if "GB" in storage and "SSD" not in storage and "RAM" not in storage:
        num = storage.replace("GB", "").replace("TB", "")
        unit = "GB" if "GB" in storage else "TB"
        return f"{num} {unit}" if alt else f"{num}{unit}"
    return storage


def _make_title(brand, model, storage, color, extra, alt: bool) -> str:
    storage_str = _fmt_storage(storage, alt)
    if storage_str:
        template = random.choice(TITLE_TEMPLATES)
        parts = template.format(brand=brand, model=model, storage=storage_str, color=color)
    else:
        plain_templates = [
            "{brand} {model} {color}", "{brand} {model}, {color}",
            "{model} {color} - {brand}", "{color} {brand} {model}",
        ]
        template = random.choice(plain_templates)
        parts = template.format(brand=brand, model=model, color=color)
    parts = " ".join(parts.split())
    if extra:
        parts = f"{parts} {extra}" if random.random() < 0.5 else f"{extra} {parts}"
    return parts


def _pick_spec_values(category: str) -> dict:
    attrs = SPEC_ATTRS.get(category, [])
    chosen_attrs = random.sample(attrs, k=min(2, len(attrs))) if attrs else []
    return {a: random.choice(SPEC_FILLERS.get(a, ["premium build"])) for a in chosen_attrs}


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


# ==========================================================================
# PAIR GENERATORS
# ==========================================================================
def generate_positive_pair_structured(category: str) -> Tuple[tuple, tuple, int]:
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

    return (
        (_next_id(), title_a, brand, desc_a),
        (_next_id(), title_b, brand, desc_b),
        1
    )


def generate_hard_negative_pair_structured(category: str) -> Tuple[tuple, tuple, int]:
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

    return (
        (_next_id(), title_a, brand, desc_a),
        (_next_id(), title_b, brand, desc_b),
        0
    )


def generate_easy_negative_pair_structured(category_a: str, category_b: str) -> Tuple[tuple, tuple, int]:
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

    return (
        (_next_id(), title_a, a_brand, desc_a),
        (_next_id(), title_b, b_brand, desc_b),
        0
    )


def generate_near_miss_model(category: str) -> Tuple[tuple, tuple, int]:
    """Same brand, storage, color -- different model."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    if len(models) < 2:
        return generate_hard_negative_pair_structured(category)
    model_a, model_b = random.sample(models, 2)
    storage = random.choice(cat["storages"])
    color, _ = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model_a, storage, color, extra, alt=False)
    title_b = _make_title(brand, model_b, storage, color, extra, alt=False)
    desc_a = _make_description(brand, model_a, _pick_spec_values(category), storage, extra, alt=False)
    desc_b = _make_description(brand, model_b, _pick_spec_values(category), storage, extra, alt=False)

    return ((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)


def generate_near_miss_storage(category: str) -> Tuple[tuple, tuple, int]:
    """Same brand, model, color -- different storage."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    storages = [s for s in cat.get("storages", []) if s is not None]
    if len(storages) < 2:
        return generate_hard_negative_pair_structured(category)
    storage_a, storage_b = random.sample(storages, 2)
    color, _ = random.choice(cat["colors"])
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model, storage_a, color, extra, alt=False)
    title_b = _make_title(brand, model, storage_b, color, extra, alt=False)
    desc_a = _make_description(brand, model, _pick_spec_values(category), storage_a, extra, alt=False)
    desc_b = _make_description(brand, model, _pick_spec_values(category), storage_b, extra, alt=False)

    return ((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)


def generate_near_miss_color(category: str) -> Tuple[tuple, tuple, int]:
    """Same brand, model, storage -- different color."""
    cat = CATEGORIES[category]
    brand, models = random.choice(cat["brands_models"])
    model = random.choice(models)
    storage = random.choice(cat["storages"])
    colors = cat.get("colors", [])
    if len(colors) < 2:
        return generate_hard_negative_pair_structured(category)
    (color_a, _), (color_b, _) = random.sample(colors, 2)
    if color_a.lower() == color_b.lower():
        return generate_hard_negative_pair_structured(category)
    extra = random.choice(cat.get("extra", [None])) if "extra" in cat else None

    title_a = _make_title(brand, model, storage, color_a, extra, alt=False)
    title_b = _make_title(brand, model, storage, color_b, extra, alt=False)
    desc_a = _make_description(brand, model, _pick_spec_values(category), storage, extra, alt=False)
    desc_b = _make_description(brand, model, _pick_spec_values(category), storage, extra, alt=False)

    return ((_next_id(), title_a, brand, desc_a), (_next_id(), title_b, brand, desc_b), 0)


# ==========================================================================
# CHUNKED GENERATION
# ==========================================================================
HEADER = [
    "product1_id", "product1_title", "product1_brand", "product1_description",
    "product2_id", "product2_title", "product2_brand", "product2_description",
    "label",
]


def _write_chunk(rows: List[tuple], out_dir: str, chunk_idx: int):
    path = os.path.join(out_dir, f"products_chunk_{chunk_idx:03d}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    n_pos = sum(1 for r in rows if r[-1] == 1)
    print(f"  Wrote chunk {chunk_idx:03d}: {len(rows)} rows ({n_pos}+ / {len(rows)-n_pos}-) -> {path}")


def generate_dataset_chunked(
    total_rows: int,
    chunk_size: int,
    out_dir: str,
    pos_ratio: float = 0.35,
    hard_neg_ratio: float = 0.30,
    easy_neg_ratio: float = 0.20,
    near_miss_ratio: float = 0.15,
):
    """
    Generates dataset in chunks to avoid memory issues.

    Mix:
      35% positive (same product, different phrasing)
      30% hard negative (same brand/category, different product)
      20% easy negative (different category)
      15% near-miss (same brand, different model/storage/color)
    """
    os.makedirs(out_dir, exist_ok=True)
    categories = list(CATEGORIES.keys())

    n_pos = int(total_rows * pos_ratio)
    n_hard_neg = int(total_rows * hard_neg_ratio)
    n_easy_neg = int(total_rows * easy_neg_ratio)
    n_near_miss = total_rows - n_pos - n_hard_neg - n_easy_neg

    # Split near-miss into sub-types
    n_near_model = n_near_miss // 3
    n_near_storage = n_near_miss // 3
    n_near_color = n_near_miss - n_near_model - n_near_storage

    generators = [
        (n_pos, lambda: generate_positive_pair_structured(random.choice(categories))),
        (n_hard_neg, lambda: generate_hard_negative_pair_structured(random.choice(categories))),
        (n_easy_neg, lambda: generate_easy_negative_pair_structured(*random.sample(categories, 2))),
        (n_near_model, lambda: generate_near_miss_model(random.choice(categories))),
        (n_near_storage, lambda: generate_near_miss_storage(random.choice(categories))),
        (n_near_color, lambda: generate_near_miss_color(random.choice(categories))),
    ]

    seen = set()
    current_chunk = []
    chunk_idx = 0
    total_generated = 0
    total_attempts = 0
    max_attempts = total_rows * 25

    def _try_add(gen_fn) -> bool:
        nonlocal total_attempts
        total_attempts += 1
        if total_attempts > max_attempts:
            return False
        side_a, side_b, label = gen_fn()
        title_a, title_b = side_a[1], side_b[1]
        key = (title_a.lower(), title_b.lower())
        key_rev = (title_b.lower(), title_a.lower())
        if key in seen or key_rev in seen or title_a.lower() == title_b.lower():
            return False
        seen.add(key)
        current_chunk.append((*side_a, *side_b, label))
        return True

    print(f"Generating {total_rows} rows in chunks of {chunk_size}...")
    print(f"  Target: {n_pos}+ | {n_hard_neg} hard- | {n_easy_neg} easy- | {n_near_miss} near-miss")

    for target_count, gen_fn in generators:
        added = 0
        while added < target_count and total_attempts < max_attempts:
            if _try_add(gen_fn):
                added += 1
                total_generated += 1
                if len(current_chunk) >= chunk_size:
                    _write_chunk(current_chunk, out_dir, chunk_idx)
                    chunk_idx += 1
                    current_chunk = []
        print(f"  -> Generated {added}/{target_count} for this category")

    # Write final partial chunk
    if current_chunk:
        _write_chunk(current_chunk, out_dir, chunk_idx)
        chunk_idx += 1

    n_pos_total = sum(1 for c in range(chunk_idx)
                      for r in _read_chunk(out_dir, c) if r[-1] == "1")
    print(f"\nDone! {chunk_idx} chunk files in {out_dir}")
    print(f"Total unique rows: {total_generated} (attempts: {total_attempts})")


def _read_chunk(out_dir: str, idx: int):
    """Helper to read back a chunk for stats."""
    path = os.path.join(out_dir, f"products_chunk_{idx:03d}.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Generate product dataset in chunks")
    parser.add_argument("--total", type=int, default=50000, help="Total rows to generate")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Rows per chunk file")
    parser.add_argument("--out-dir", type=str, default="./data_chunks", help="Output directory")
    parser.add_argument("--pos-ratio", type=float, default=0.35)
    parser.add_argument("--hard-neg-ratio", type=float, default=0.30)
    parser.add_argument("--easy-neg-ratio", type=float, default=0.20)
    parser.add_argument("--near-miss-ratio", type=float, default=0.15)
    args = parser.parse_args()

    assert abs(args.pos_ratio + args.hard_neg_ratio + args.easy_neg_ratio + args.near_miss_ratio - 1.0) < 0.001,         "Ratios must sum to 1.0"

    generate_dataset_chunked(
        total_rows=args.total,
        chunk_size=args.chunk_size,
        out_dir=args.out_dir,
        pos_ratio=args.pos_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        easy_neg_ratio=args.easy_neg_ratio,
        near_miss_ratio=args.near_miss_ratio,
    )


if __name__ == "__main__":
    main()
