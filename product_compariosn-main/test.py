from exact_match.inference import ProductComparer
comparer = ProductComparer()
test_cases = [
    # (label, title_a, brand_a, desc_a, title_b, brand_b, desc_b, expected)

    ("1. Same laptop, different retailer wording", "Lenovo ThinkPad X1 Carbon Gen 10 16GB 512GB", "Lenovo",
     "16GB RAM, 512GB SSD, Intel i7", "ThinkPad X1 Carbon (Gen 10) with 16GB memory and 512GB storage",
     "Lenovo", "i7 processor, 16GB RAM, 512GB storage", "Same"),

    ("2. Same TV, different screen size", "Samsung QN90B 55 inch 4K Smart TV", "Samsung",
     "Neo QLED, 120Hz, HDR", "Samsung QN90B 65 inch 4K Smart TV", "Samsung",
     "65-inch Neo QLED, 120Hz, HDR", "Different"),

    ("3. Same headphones, different color", "Sony WH-1000XM5 Black", "Sony",
     "Wireless noise cancelling, LDAC", "Sony WH-1000XM5 Silver", "Sony",
     "Wireless noise cancelling, LDAC", "Different"),

    ("4. Same phone, single vs dual SIM", "Samsung Galaxy S24 128GB Dual SIM", "Samsung",
     "Snapdragon 8 Gen 3, 8GB RAM", "Samsung Galaxy S24 128GB Single SIM", "Samsung",
     "Snapdragon 8 Gen 3, 8GB RAM", "Different"),

    ("5. iPad Pro different chip generation", "Apple iPad Pro 11-inch M2 128GB", "Apple",
     "Liquid Retina, Face ID", "Apple iPad Pro 11-inch M1 128GB", "Apple",
     "Liquid Retina, Face ID", "Different"),

    ("6. Same book, different cover (hardcover vs paperback)", "The Hobbit (paperback)", "J.R.R. Tolkien",
     "Fantasy classic", "The Hobbit (hardcover)", "J.R.R. Tolkien",
     "Fantasy classic", "Same"),

    ("7. Same book, different edition", "1984 (2nd Edition)", "George Orwell",
     "Dystopian classic", "1984 (3rd Edition)", "George Orwell",
     "Dystopian classic with new foreword", "Different"),

    ("8. Same game, different platform", "FIFA 23 PS5", "EA Sports",
     "Football simulation", "FIFA 23 Xbox Series X", "EA Sports",
     "Football simulation", "Different"),

    ("9. Same external drive, different capacity", "WD My Passport 1TB", "Western Digital",
     "USB 3.0, portable", "WD My Passport 2TB", "Western Digital",
     "USB 3.0, portable", "Different"),

    ("10. Same brand, different laptop series", "Dell XPS 13", "Dell",
     "Ultrabook, 13-inch", "Dell Latitude 5420", "Dell",
     "Business laptop, 14-inch", "Different"),

    ("11. Same monitor, full name vs abbreviation", "Dell UltraSharp U2723QE 4K", "Dell",
     "IPS, USB-C hub", "Dell U2723QE", "Dell",
     "27-inch 4K IPS, USB-C", "Same"),

    ("12. Same phone, unlocked vs carrier-locked", "Apple iPhone 15 Pro Unlocked", "Apple",
     "A17 Pro, 128GB", "Apple iPhone 15 Pro AT&T Locked", "Apple",
     "A17 Pro, 128GB", "Same"),

    ("13. Different brand, similar product", "Apple Watch Series 9", "Apple",
     "Smartwatch, health tracking", "Samsung Galaxy Watch6", "Samsung",
     "Smartwatch, health tracking", "Different"),

    ("14. Same camera, different bundle", "Canon EOS R5 Body", "Canon",
     "45MP, 8K video", "Canon EOS R5 with 24-105mm Lens", "Canon",
     "45MP, 8K video, kit lens", "Different"),

    ("15. Same phone, new vs renewed", "Apple iPhone 12 (New)", "Apple",
     "A14, 64GB", "Apple iPhone 12 (Renewed)", "Apple",
     "A14, 64GB, certified refurbished", "Different"),

    ("16. Same laptop, different OS pre-installed", "Dell XPS 13 Windows 11", "Dell",
     "Core i7, 16GB", "Dell XPS 13 Ubuntu", "Dell",
     "Core i7, 16GB, Linux", "Different"),

    ("17. Same tablet, different storage", "Samsung Galaxy Tab S8 128GB", "Samsung",
     "11-inch, 120Hz", "Samsung Galaxy Tab S8 256GB", "Samsung",
     "11-inch, 120Hz", "Different"),

    ("18. Same shoe, different width", "Nike Air Max 270 UK 9 D", "Nike",
     "Men's running shoe", "Nike Air Max 270 UK 9 2E", "Nike",
     "Men's running shoe, wide fit", "Different"),

    ("19. Same coffee maker, different color", "Keurig K-Elite Black", "Keurig",
     "Single serve, strong brew", "Keurig K-Elite Red", "Keurig",
     "Single serve, strong brew", "Different"),

    ("20. Same TV, different description wording", "Samsung 55-inch Smart TV QLED", "Samsung",
     "4K, HDR, 120Hz", "Samsung QLED 4K TV 55\"", "Samsung",
     "120Hz, HDR, Smart", "Same"),

    ("21. Same brand with 'Inc.'", "Apple MacBook Pro 14-inch", "Apple",
     "M3, 16GB, 512GB", "Apple Inc. MacBook Pro 14\"", "Apple",
     "M3, 16GB, 512GB SSD", "Same"),

    ("22. Different generation Kindle", "Kindle Paperwhite 5", "Amazon",
     "6.8\" display, waterproof", "Kindle Paperwhite 4", "Amazon",
     "6\" display, waterproof", "Different"),

    ("23. Different external SSD models (with/without touch)", "Samsung T7 Portable 1TB", "Samsung",
     "USB 3.2, 1050MB/s", "Samsung T7 Touch 1TB", "Samsung",
     "USB 3.2, fingerprint security", "Different"),

    ("24. Same router, different hardware version", "TP-Link Archer AX50", "TP-Link",
     "Wi-Fi 6, dual-band", "TP-Link Archer AX50 v2", "TP-Link",
     "Wi-Fi 6, dual-band", "Different"),

    ("25. Different blender models", "NutriBullet Pro 900", "NutriBullet",
     "900W, 32oz cup", "NutriBullet Pro 1000", "NutriBullet",
     "1000W, 32oz cup", "Different"),

    ("26. Same vacuum, different accessory bundle", "Dyson V15 with crevice tool", "Dyson",
     "Digital motor, laser", "Dyson V15 with pet hair tool", "Dyson",
     "Digital motor, laser", "Different"),

    ("27. Same toothbrush, different color (set as Different)", "Philips Sonicare 6100 Black", "Philips",
     "3 intensities, timer", "Philips Sonicare 6100 White", "Philips",
     "3 intensities, timer", "Different"),

    ("28. Same watch, different strap", "Apple Watch Series 8 with Sport Band", "Apple",
     "Always-on display, ECG", "Apple Watch Series 8 with Milanese Loop", "Apple",
     "Always-on display, ECG", "Different"),

    ("29. Same LEGO set, different packaging wording", "LEGO Millennium Falcon 75192", "LEGO",
     "Star Wars, 7541 pieces", "LEGO Star Wars Millennium Falcon (75192)", "LEGO",
     "Includes minifigures, 7541 pieces", "Same"),

    ("30. Same printer, with/without extra ink", "HP Envy 6055", "HP",
     "Wireless, print/scan/copy", "HP Envy 6055 with extra ink cartridges", "HP",
     "Wireless, print/scan/copy", "Same"),

    ("31. Same phone, different network tech", "Apple iPhone 14 Pro GSM", "Apple",
     "A16, 6.1\"", "Apple iPhone 14 Pro CDMA", "Apple",
     "A16, 6.1\"", "Same"),

    ("32. Same laptop, different RAM", "Dell XPS 13 8GB RAM", "Dell",
     "Core i5, 256GB SSD", "Dell XPS 13 16GB RAM", "Dell",
     "Core i5, 256GB SSD", "Different"),

    ("33. Same product, different model year", "Apple MacBook Air 2023", "Apple",
     "M2, 13.6\"", "Apple MacBook Air 2024", "Apple",
     "M3, 13.6\"", "Different"),

    ("34. Same perfume, different bottle size", "Chanel No.5 50ml", "Chanel",
     "Floral aldehyde", "Chanel No.5 100ml", "Chanel",
     "Floral aldehyde", "Different"),

    ("35. Same book, different title spelling", "1984 by George Orwell", "Penguin",
     "Classic dystopian", "Nineteen Eighty-Four", "Penguin",
     "Classic dystopian", "Same"),

    ("36. Same game, standard vs GOTY", "The Witcher 3 Standard", "CD Projekt Red",
     "RPG open world", "The Witcher 3 Game of the Year", "CD Projekt Red",
     "Includes expansions", "Different"),

    ("37. Different monitor models (similar names)", "ASUS ROG Swift PG279Q", "ASUS",
     "165Hz, IPS", "ASUS ROG Swift PG279QM", "ASUS",
     "240Hz, IPS", "Different"),

    ("38. Same CPU, retail vs tray", "Intel Core i9-13900K Retail", "Intel",
     "24 cores, 5.8GHz", "Intel Core i9-13900K Tray", "Intel",
     "24 cores, 5.8GHz", "Same"),

    ("39. Same TV, US vs EU region", "Sony X90K 55\" US", "Sony",
     "4K HDR, Google TV", "Sony X90K 55\" EU", "Sony",
     "4K HDR, Google TV", "Same"),

    ("40. Same headphones, different feature keywords", "Sony WH-1000XM5 Noise Cancelling", "Sony",
     "Wireless, LDAC", "Sony WH-1000XM5 Wireless Headphones", "Sony",
     "Noise cancelling, LDAC", "Same"),

    ("41. Same appliance, different voltage", "Ninja Foodi 220V", "Ninja",
     "Pressure cooker, air fryer", "Ninja Foodi 110V", "Ninja",
     "Pressure cooker, air fryer", "Different"),

    ("42. Same yoga mat, different thickness", "Manduka PRO 6mm", "Manduka",
     "Non-slip, eco-friendly", "Manduka PRO 4mm", "Manduka",
     "Non-slip, eco-friendly", "Different"),

    ("43. Same shoe, different gender (men's vs women's)", "Nike Air Max 270 Men's", "Nike",
     "UK 9, running", "Nike Air Max 270 Women's", "Nike",
     "UK 7, running", "Different"),

    ("44. Same sunglasses, different frame color", "Ray-Ban RB2132 Black", "Ray-Ban",
     "Wayfarer, polarized", "Ray-Ban RB2132 Tortoise", "Ray-Ban",
     "Wayfarer, polarized", "Different"),

    ("45. Same monitor, curved vs flat (different model)", "Samsung Odyssey G5 Curved 27\"", "Samsung",
     "144Hz, VA", "Samsung Odyssey G5 Flat 27\"", "Samsung",
     "144Hz, IPS", "Different"),

    ("46. Same wireless charger, different cable length", "Belkin BOOST CHARGE 15W", "Belkin",
     "Qi, with 1m cable", "Belkin BOOST CHARGE 15W", "Belkin",
     "Qi, with 2m cable", "Same"),

    ("47. Same coffee beans, different roast", "Starbucks French Roast", "Starbucks",
     "Dark roast, whole bean", "Starbucks Dark Roast", "Starbucks",
     "Dark roast, whole bean", "Different"),

    ("48. Same laptop, touch vs non-touch", "Dell XPS 13 Touch", "Dell",
     "Core i7, 16GB", "Dell XPS 13 Non-Touch", "Dell",
     "Core i7, 16GB", "Different"),

    ("49. Same smart speaker, different color fabric", "Sonos One Black", "Sonos",
     "Voice control, Wi-Fi", "Sonos One White", "Sonos",
     "Voice control, Wi-Fi", "Different"),

    ("50. Same product, different brand name abbreviation", "Microsoft Surface Pro 9", "Microsoft",
     "Intel Evo, 13\"", "MS Surface Pro 9", "Microsoft",
     "Intel Evo, 13\"", "Same"),
]

correct = 0
for name, ta, ba, da, tb, bb, db, expected in test_cases:
    r = comparer.compare(title_a=ta, brand_a=ba, description_a=da, title_b=tb, brand_b=bb, description_b=db)
    got = "Same" if r.label == 1 else "Different"
    status = "✅" if got == expected else "❌"
    if got == expected:
        correct += 1
    print(f"{status} {name}: {r.similarity_score:.1f}% -> {got} (expected {expected})")

print(f"\n{correct}/{len(test_cases)} correct")