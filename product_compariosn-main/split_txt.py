from pathlib import Path

input_file = r"C:\Users\manik\Downloads\product_v1\product_compariosn-main\data\descriptions.txt\descriptions.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
part_size = total // 3

parts = [
    lines[:part_size],
    lines[part_size:2 * part_size],
    lines[2 * part_size:]
]

for i, part in enumerate(parts, start=1):
    with open(f"text_part_{i}.txt", "w", encoding="utf-8") as out:
        out.writelines(part)

print("Done!")
print(f"Total lines: {total}")
print(f"Part 1: {len(parts[0])} lines")
print(f"Part 2: {len(parts[1])} lines")
print(f"Part 3: {len(parts[2])} lines")