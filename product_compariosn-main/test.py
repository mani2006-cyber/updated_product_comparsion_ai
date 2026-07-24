import sys, gc, csv

sys.path.insert(0, ".")

from filter_and_test_large_dataset import (
    _parse_id_value_file,
    _parse_categories_file,
    _parse_description_file
)

titles = _parse_id_value_file(
    r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\data\titles.txt\titles.txt"
)

categories = _parse_categories_file(
    r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\data\categories.txt\categories.txt"
)

top_categories = {
    pid: (paths[0].split(",")[0].strip() if paths else "UNKNOWN")
    for pid, paths in categories.items()
}

del categories
gc.collect()

brands = _parse_id_value_file(
    r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\data\brands.txt\brands.txt"
)

with open(
    r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\data\full_joined_catalog.csv",
    "w",
    newline="",
    encoding="utf-8"
) as out_f:

    writer = csv.writer(out_f)
    writer.writerow(["id", "title", "brand", "description", "category"])

    for part_path in [
        r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\text_part_1.txt",
        r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\text_part_2.txt",
        r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\text_part_3.txt",
    ]:

        part_descs = _parse_description_file(part_path)

        for pid, desc in part_descs.items():
            if pid in titles and pid in top_categories:
                writer.writerow([
                    pid,
                    titles[pid],
                    brands.get(pid, ""),
                    desc,
                    top_categories[pid]
                ])

        del part_descs
        gc.collect()

print("Done!")