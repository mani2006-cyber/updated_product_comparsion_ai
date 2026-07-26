import config
from exact_match.inference import ProductComparer

comparer = ProductComparer(model_dir=config.TRAINED_MODEL_DIR)

diagnostic_tests = [

# ---------- EXACT MATCH ----------

dict(
title_a="boAt Airdopes 141",
brand_a="boAt",
description_a="42 hour playtime ENx tech Bluetooth 5.3",

title_b="boAt Airdopes 141",
brand_b="boAt",
description_b="42 hour playtime ENx tech Bluetooth 5.3"
),

dict(
title_a="Apple iPhone 15 Pro 256GB Blue",
brand_a="Apple",
description_a="A17 Pro chip",

title_b="Apple iPhone 15 Pro 256GB Blue",
brand_b="Apple",
description_b="A17 Pro chip"
),

dict(
title_a="Samsung Galaxy S24 Ultra",
brand_a="Samsung",
description_a="12GB RAM 512GB",

title_b="Samsung Galaxy S24 Ultra",
brand_b="Samsung",
description_b="12GB RAM 512GB"
),

dict(
title_a="Sony WH-1000XM5",
brand_a="Sony",
description_a="Noise Cancelling",

title_b="Sony WH-1000XM5",
brand_b="Sony",
description_b="Noise Cancelling"
),

dict(
title_a="Dell Inspiron 15 3530",
brand_a="Dell",
description_a="Core i5 16GB RAM",

title_b="Dell Inspiron 15 3530",
brand_b="Dell",
description_b="Core i5 16GB RAM"
),

# ---------- SAME PRODUCT DIFFERENT VARIANT ----------

dict(
title_a="Apple iPhone 15 Pro 128GB",
brand_a="Apple",
description_a="Black",

title_b="Apple iPhone 15 Pro 256GB",
brand_b="Apple",
description_b="Black"
),

dict(
title_a="boAt Airdopes 141 Black",
brand_a="boAt",
description_a="42 hour battery",

title_b="boAt Airdopes 141 White",
brand_b="boAt",
description_b="42 hour battery"
),

dict(
title_a="Samsung Galaxy S24 Ultra 256GB",
brand_a="Samsung",
description_a="Titanium Black",

title_b="Samsung Galaxy S24 Ultra 512GB",
brand_b="Samsung",
description_b="Titanium Black"
),

dict(
title_a="OnePlus Nord CE4 8GB",
brand_a="OnePlus",
description_a="128GB",

title_b="OnePlus Nord CE4 12GB",
brand_b="OnePlus",
description_b="256GB"
),

dict(
title_a="HP Victus 15 RTX4050",
brand_a="HP",
description_a="16GB RAM",

title_b="HP Victus 15 RTX4060",
brand_b="HP",
description_b="16GB RAM"
),

# ---------- SIMILAR ALTERNATIVE ----------

dict(
title_a="boAt Airdopes 141",
brand_a="boAt",
description_a="42 hour battery",

title_b="Noise Buds VS104",
brand_b="Noise",
description_b="45 hour battery"
),

dict(
title_a="Samsung Galaxy S24",
brand_a="Samsung",
description_a="8GB RAM",

title_b="Google Pixel 9",
brand_b="Google",
description_b="12GB RAM"
),

dict(
title_a="Apple MacBook Air M3",
brand_a="Apple",
description_a="13 inch",

title_b="Dell XPS 13",
brand_b="Dell",
description_b="13 inch"
),

dict(
title_a="Sony WH1000XM5",
brand_a="Sony",
description_a="ANC",

title_b="Bose QuietComfort Ultra",
brand_b="Bose",
description_b="ANC"
),

dict(
title_a="Lenovo Legion 5",
brand_a="Lenovo",
description_a="RTX4060",

title_b="ASUS ROG Strix G16",
brand_b="ASUS",
description_b="RTX4060"
),

# ---------- WEAKLY SIMILAR ----------

dict(
title_a="Samsung Smart TV",
brand_a="Samsung",
description_a="55 inch",

title_b="LG Smart TV",
brand_b="LG",
description_b="55 inch"
),

dict(
title_a="Apple Watch Series 10",
brand_a="Apple",
description_a="GPS",

title_b="Samsung Galaxy Watch 7",
brand_b="Samsung",
description_b="GPS"
),

dict(
title_a="Canon DSLR Camera",
brand_a="Canon",
description_a="24MP",

title_b="Nikon DSLR Camera",
brand_b="Nikon",
description_b="24MP"
),

dict(
title_a="HP Pavilion Laptop",
brand_a="HP",
description_a="Ryzen 7",

title_b="Acer Aspire Laptop",
brand_b="Acer",
description_b="Ryzen 7"
),

dict(
title_a="Realme Narzo 70",
brand_a="Realme",
description_a="5G",

title_b="Redmi Note 14",
brand_b="Xiaomi",
description_b="5G"
),

# ---------- UNRELATED ----------

dict(
title_a="Apple iPhone 15",
brand_a="Apple",
description_a="Smartphone",

title_b="Nike Air Max Shoes",
brand_b="Nike",
description_b="Running shoes"
),

dict(
title_a="Samsung Refrigerator",
brand_a="Samsung",
description_a="Double Door",

title_b="Dell Inspiron Laptop",
brand_b="Dell",
description_b="Core i7"
),

dict(
title_a="Sony Camera",
brand_a="Sony",
description_a="Mirrorless",

title_b="boAt Smartwatch",
brand_b="boAt",
description_b="AMOLED"
),

dict(
title_a="LG Washing Machine",
brand_a="LG",
description_a="Front Load",

title_b="Apple MacBook Pro",
brand_b="Apple",
description_b="M4 Pro"
),

dict(
title_a="Puma Backpack",
brand_a="Puma",
description_a="30L",

title_b="Samsung SSD",
brand_b="Samsung",
description_b="1TB NVMe"
),

# ---------- TYPO TESTS ----------

dict(
title_a="boAt Airdopes 141",
brand_a="boAt",
description_a="42 hour",

title_b="Boat Airdopse 141",
brand_b="Boat",
description_b="42 hr"
),

dict(
title_a="Samsung Galaxy S24",
brand_a="Samsung",
description_a="256GB",

title_b="Samsng Galaxy S24",
brand_b="Samsung",
description_b="256 GB"
),

dict(
title_a="iPhone 15 Pro",
brand_a="Apple",
description_a="A17",

title_b="Iphone15 Pro",
brand_b="Apple",
description_b="A17 Pro"
),

dict(
title_a="Dell Inspiron",
brand_a="Dell",
description_a="Core i5",

title_b="Del Inspiron",
brand_b="Dell",
description_b="Intel i5"
),

dict(
title_a="Sony WH1000XM5",
brand_a="Sony",
description_a="ANC",

title_b="Sony WH100XM5",
brand_b="Sony",
description_b="Noise Cancel"
),

# ---------- BRAND CONFUSION ----------

dict(
title_a="Galaxy S24",
brand_a="Samsung",
description_a="",

title_b="Galaxy S24",
brand_b="FakeBrand",
description_b=""
),

dict(
title_a="iPhone 15",
brand_a="Apple",
description_a="",

title_b="iPhone 15",
brand_b="Clone",
description_b=""
),

dict(
title_a="AirPods Pro",
brand_a="Apple",
description_a="",

title_b="AirPods Pro",
brand_b="Boat",
description_b=""
),

dict(
title_a="ROG Phone 9",
brand_a="ASUS",
description_a="",

title_b="ROG Phone 9",
brand_b="Lenovo",
description_b=""
),

dict(
title_a="ThinkPad X1",
brand_a="Lenovo",
description_a="",

title_b="ThinkPad X1",
brand_b="Dell",
description_b=""
)

]

for i, tc in enumerate(diagnostic_tests, 1):
    result = comparer.compare(**tc)

    print("=" * 80)
    print(f"TEST {i}")
    print(tc["title_a"])
    print("VS")
    print(tc["title_b"])
    print("-" * 80)
    print("Prediction :", result.relationship)
    print("Score      :", f"{result.similarity_score:.2f}%")
    print("Probabilities")
    print(result.all_probabilities)
    print()