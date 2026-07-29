import config
from exact_match.inference import ProductComparer

comparer = ProductComparer(model_dir=config.TRAINED_MODEL_DIR)

audio_stress_test = [
    dict(
        title_a="GOBOULT Z40 Pro TWS Earbuds",
        brand_a="GOBOULT",
        description_a="100 hour playtime, quad-mic ENC",
        title_b="pTron Bassbuds Astra TWS Earbuds",
        brand_b="pTron",
        description_b="34 hour playtime, Bluetooth 5.3"
    ),
    dict(
        title_a="boAt Airdopes 141",
        brand_a="boAt",
        description_a="42 hour playtime, ENx tech",
        title_b="Zebronics Sound Bomb 4",
        brand_b="Zebronics",
        description_b="20 hour playtime, 13mm drivers"
    ),
    dict(
        title_a="Noise Buds VS404",
        brand_a="Noise",
        description_a="AI-ENx, low latency",
        title_b="pTron Bassbuds Sports",
        brand_b="pTron",
        description_b="32 hour playtime, secure fit"
    ),
    dict(
        title_a="boAt Airdopes 300",
        brand_a="boAt",
        description_a="TWS earbuds, 50 hour battery",
        title_b="Samsung Galaxy S24 Ultra",
        brand_b="Samsung",
        description_b="Snapdragon 8 Gen 3, 200MP camera"
    ),
    dict(
        title_a="GOBOULT Mustang Stallion Smartwatch",
        brand_a="GOBOULT",
        description_a="sporty design, all-day battery",
        title_b="Apple iPhone 16 Pro",
        brand_b="Apple",
        description_b="A18 Pro chip, 48MP camera"
    ),
]

for i, tc in enumerate(audio_stress_test, 1):
    result = comparer.compare(**tc)
    print(f"--- Test {i}: {tc['title_a'][:40]} vs {tc['title_b'][:40]} ---")
    print(result)
    print(result.relationship)
    print(result.similarity_score)
    print(result.all_probabilities)
    print()