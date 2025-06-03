# import os
# import csv
# import random

# DATASET_DIR = 'dataset'
# OUTPUT_DIR = 'data'

# TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
# VAL_CSV = os.path.join(OUTPUT_DIR, 'val.csv')
# TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')

# def extract_pairs_from_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = [line.strip() for line in f if line.strip()]
    
#     pairs = []
#     for i in range(len(lines) - 1):
#         input_masra = lines[i]
#         output_masra = lines[i + 1]
#         if input_masra != output_masra:
#             pairs.append((input_masra, output_masra))
#     return pairs

# def main():
#     all_pairs = []
#     for filename in os.listdir(DATASET_DIR):
#         if filename.endswith('.txt'):
#             filepath = os.path.join(DATASET_DIR, filename)
#             pairs = extract_pairs_from_file(filepath)
#             all_pairs.extend(pairs)

#     if not all_pairs:
#         print("⚠️ No valid masra pairs found.")
#         return

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Step 1: Write train.csv (100% original order)
#     with open(TRAIN_CSV, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['input', 'output'])
#         writer.writerows(all_pairs)

#     # Step 2: Create shuffled copy of all_pairs
#     shuffled = all_pairs[:]
#     random.shuffle(shuffled)

#     val_size = int(len(shuffled) * 0.20)
#     test_size = int(len(shuffled) * 0.20)

#     val_pairs = shuffled[:val_size]
#     test_pairs = shuffled[val_size:val_size + test_size]

#     # Step 3: Write val.csv
#     with open(VAL_CSV, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['input', 'output'])
#         writer.writerows(val_pairs)

#     # Step 4: Write test.csv
#     with open(TEST_CSV, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['input', 'output'])
#         writer.writerows(test_pairs)

#     print(f"✅ train.csv → {len(all_pairs)} pairs (no shuffle)")
#     print(f"✅ val.csv   → {len(val_pairs)} pairs (shuffled 20%)")
#     print(f"✅ test.csv  → {len(test_pairs)} pairs (shuffled 20%)")

# if __name__ == '__main__':
#     main()

import os
import csv
import random
import unicodedata

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'data'

TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
VAL_CSV = os.path.join(OUTPUT_DIR, 'val.csv')
TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')

def normalize(text):
    # Normalize Unicode (NFC) and strip leading/trailing spaces
    return unicodedata.normalize('NFC', text.strip())

def extract_pairs_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [normalize(line) for line in f if line.strip()]
    
    pairs = []
    for i in range(len(lines) - 1):
        input_masra = lines[i]
        output_masra = lines[i + 1]
        if input_masra and output_masra:
            pairs.append((input_masra, output_masra))
    return pairs

def clean_pairs(pairs):
    seen = set()
    cleaned = []
    for inp, out in pairs:
        inp = normalize(inp)
        out = normalize(out)
        if inp != out:
            key = (inp, out)
            if key not in seen:
                seen.add(key)
                cleaned.append(key)
    return cleaned

def write_csv(path, pairs):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'output'])
        writer.writerows(pairs)

def main():
    all_pairs = []
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(DATASET_DIR, filename)
            pairs = extract_pairs_from_file(filepath)
            all_pairs.extend(pairs)

    if not all_pairs:
        print("⚠️ No valid masra pairs found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Write raw train.csv (100% original order)
    write_csv(TRAIN_CSV, all_pairs)

    # Step 2: Shuffle and split for val/test
    shuffled = all_pairs[:]
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * 0.20)
    test_size = int(len(shuffled) * 0.20)

    val_pairs = shuffled[:val_size]
    test_pairs = shuffled[val_size:val_size + test_size]

    write_csv(VAL_CSV, val_pairs)
    write_csv(TEST_CSV, test_pairs)

    print(f"Raw train.csv → {len(all_pairs)} pairs")
    print(f"Raw val.csv   → {len(val_pairs)} pairs")
    print(f"Raw test.csv  → {len(test_pairs)} pairs")

    # Step 3: Clean all three CSVs (removing duplicates and input==output)
    for file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            raw_pairs = [(normalize(row['input']), normalize(row['output'])) for row in reader]

        cleaned = clean_pairs(raw_pairs)
        write_csv(file, cleaned)
        print(f"🧼 Cleaned {file} → {len(cleaned)} pairs")

if __name__ == '__main__':
    main()
