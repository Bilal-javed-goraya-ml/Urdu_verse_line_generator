# import os
# import csv
# import random
# import unicodedata
# from collections import defaultdict

# DATASET_DIR = 'dataset'
# OUTPUT_DIR = 'data'

# TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
# VAL_CSV = os.path.join(OUTPUT_DIR, 'val.csv')
# TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')

# def normalize(text):
#     # Normalize Unicode (NFC) and strip leading/trailing spaces
#     return unicodedata.normalize('NFC', text.strip())

# def extract_pairs_from_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = [normalize(line) for line in f if line.strip()]
    
#     pairs = []
#     for i in range(len(lines) - 1):
#         input_masra = lines[i]
#         output_masra = lines[i + 1]
#         if input_masra and output_masra:
#             pairs.append((input_masra, output_masra))
#     return pairs

# # def clean_pairs(pairs):
# #     seen = set()
# #     cleaned = []
# #     for inp, out in pairs:
# #         inp = normalize(inp)
# #         out = normalize(out)
# #         if inp != out:
# #             key = (inp, out)
# #             if key not in seen:
# #                 seen.add(key)
# #                 cleaned.append(key)
# #     return cleaned


# # No input == output
# # No reverse pairs
# # No direct or indirect cycles like a→b→c→a
# def clean_pairs(pairs):
#     seen_pairs = set()
#     graph = defaultdict(set)
#     cleaned = []

#     def has_cycle(start, target, visited):
#         if start == target:
#             return True
#         visited.add(start)
#         for neighbor in graph[start]:
#             if neighbor not in visited:
#                 if has_cycle(neighbor, target, visited):
#                     return True
#         return False

#     for inp, out in pairs:
#         inp = normalize(inp)
#         out = normalize(out)

#         if inp == out:
#             continue

#         pair = (inp, out)
#         reverse_pair = (out, inp)

#         # Check for reverse and exact pair
#         if pair in seen_pairs or reverse_pair in seen_pairs:
#             continue

#         # Check for cycle (i.e., out eventually leads back to inp)
#         if has_cycle(out, inp, set()):
#             continue

#         # Passed all checks
#         seen_pairs.add(pair)
#         graph[inp].add(out)
#         cleaned.append(pair)

#     return cleaned


# def write_csv(path, pairs):
#     with open(path, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['input', 'output'])
#         writer.writerows(pairs)

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

#     # Step 1: Write raw train.csv (100% original order)
#     write_csv(TRAIN_CSV, all_pairs)

#     # Step 2: Shuffle and split for val/test
#     shuffled = all_pairs[:]
#     random.shuffle(shuffled)

#     val_size = int(len(shuffled) * 0.20)
#     test_size = int(len(shuffled) * 0.20)

#     val_pairs = shuffled[:val_size]
#     test_pairs = shuffled[val_size:val_size + test_size]

#     write_csv(VAL_CSV, val_pairs)
#     write_csv(TEST_CSV, test_pairs)

#     print(f"Raw train.csv → {len(all_pairs)} pairs")
#     print(f"Raw val.csv   → {len(val_pairs)} pairs")
#     print(f"Raw test.csv  → {len(test_pairs)} pairs")

#     # Step 3: Clean all three CSVs (removing duplicates and input==output)
#     for file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
#         with open(file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             raw_pairs = [(normalize(row['input']), normalize(row['output'])) for row in reader]

#         cleaned = clean_pairs(raw_pairs)
#         write_csv(file, cleaned)
#         print(f"🧼 Cleaned {file} → {len(cleaned)} pairs")

# if __name__ == '__main__':
#     main()

# import os
# import csv
# import random
# import unicodedata
# from collections import defaultdict

# DATASET_DIR = 'dataset'
# OUTPUT_DIR = 'data'

# TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
# VAL_CSV = os.path.join(OUTPUT_DIR, 'val.csv')
# TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')

# KHATAM_TOKEN = 'ختم'


# def normalize(text):
#     return unicodedata.normalize('NFC', text.strip())


# def extract_pairs_from_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = [normalize(line) for line in f if line.strip()]

#     pairs = []
#     used_inputs = set()

#     for i in range(len(lines) - 1):
#         inp = lines[i]
#         out = lines[i + 1]

#         # Ensure input not seen before
#         if inp not in used_inputs and inp != out:
#             pairs.append((inp, out))
#             used_inputs.add(inp)

#     # Add last masra → "ختم"
#     if lines:
#         last_line = lines[-1]
#         if last_line not in used_inputs:
#             pairs.append((last_line, KHATAM_TOKEN))

#     return pairs


# def has_cycle(graph, start, target, visited):
#     if start == target:
#         return True
#     visited.add(start)
#     for neighbor in graph[start]:
#         if neighbor not in visited:
#             if has_cycle(graph, neighbor, target, visited):
#                 return True
#     return False


# def clean_pairs(pairs):
#     seen_pairs = set()
#     graph = defaultdict(set)
#     cleaned = []

#     inputs_used = set()
#     outputs_used = set()

#     for inp, out in pairs:
#         inp = normalize(inp)
#         out = normalize(out)

#         if inp == out:
#             continue

#         pair = (inp, out)
#         reverse_pair = (out, inp)

#         if pair in seen_pairs or reverse_pair in seen_pairs:
#             continue

#         # Prevent multiple uses of same input or output (except 'ختم' in output)
#         if inp in inputs_used or (out in outputs_used and out != KHATAM_TOKEN):
#             continue

#         if has_cycle(graph, out, inp, set()):
#             continue

#         # Passed all checks
#         seen_pairs.add(pair)
#         graph[inp].add(out)
#         cleaned.append(pair)
#         inputs_used.add(inp)
#         if out != KHATAM_TOKEN:
#             outputs_used.add(out)

#     return cleaned


# def write_csv(path, pairs):
#     with open(path, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['input', 'output'])
#         writer.writerows(pairs)


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

#     # Step 1: Write raw train.csv (original order)
#     write_csv(TRAIN_CSV, all_pairs)

#     # Step 2: Shuffle and split for val/test
#     shuffled = all_pairs[:]
#     random.shuffle(shuffled)

#     val_size = int(len(shuffled) * 0.20)
#     test_size = int(len(shuffled) * 0.20)

#     val_pairs = shuffled[:val_size]
#     test_pairs = shuffled[val_size:val_size + test_size]

#     write_csv(VAL_CSV, val_pairs)
#     write_csv(TEST_CSV, test_pairs)

#     print(f"Raw train.csv → {len(all_pairs)} pairs")
#     print(f"Raw val.csv   → {len(val_pairs)} pairs")
#     print(f"Raw test.csv  → {len(test_pairs)} pairs")

#     # Step 3: Clean all three CSVs
#     for file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
#         with open(file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             raw_pairs = [(normalize(row['input']), normalize(row['output'])) for row in reader]

#         cleaned = clean_pairs(raw_pairs)
#         write_csv(file, cleaned)
#         print(f"🧼 Cleaned {file} → {len(cleaned)} pairs")


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

KHATAM_TOKEN = 'ختم'


def normalize(text):
    return unicodedata.normalize('NFC', text.strip())


# def extract_pairs_from_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = [normalize(line) for line in f if line.strip()]

#     # Skip files with duplicate masras
#     masra_counts = {}
#     for line in lines:
#         masra_counts[line] = masra_counts.get(line, 0) + 1
#     if any(count > 1 for count in masra_counts.values()):
#         return []

#     pairs = []
#     used_inputs = set()
#     used_outputs = set()

#     for i in range(len(lines) - 1):
#         inp = lines[i]
#         out = lines[i + 1]

#         if inp in used_inputs or out in used_outputs:
#             continue
#         if inp == out:
#             continue

#         pairs.append((inp, out))
#         used_inputs.add(inp)
#         used_outputs.add(out)

#     # Add last masra → ختم if it wasn't used as input
#     if lines:
#         last = lines[-1]
#         if last not in used_inputs:
#             pairs.append((last, KHATAM_TOKEN))
#             used_inputs.add(last)

#     return pairs


def extract_pairs_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [normalize(line) for line in f if line.strip()]

    # Remove duplicates masras completely (everywhere in file)
    masra_counts = {}
    for line in lines:
        masra_counts[line] = masra_counts.get(line, 0) + 1

    unique_lines = [line for line in lines if masra_counts[line] == 1]

    pairs = []
    used_inputs = set()
    used_outputs = set()

    for i in range(len(unique_lines) - 1):
        inp = unique_lines[i]
        out = unique_lines[i + 1]

        if inp in used_inputs or out in used_outputs:
            continue
        if inp == out:
            continue

        pairs.append((inp, out))
        used_inputs.add(inp)
        used_outputs.add(out)

    # Add last masra → ختم if not already input
    if unique_lines:
        last = unique_lines[-1]
        if last not in used_inputs:
            pairs.append((last, KHATAM_TOKEN))
            used_inputs.add(last)

    return pairs


def write_csv(path, pairs):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'output'])
        writer.writerows(pairs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pairs = []
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(DATASET_DIR, filename)
            pairs = extract_pairs_from_file(filepath)
            all_pairs.extend(pairs)

    if not all_pairs:
        print("⚠️ No valid masra pairs found.")
        return

    write_csv(TRAIN_CSV, all_pairs)

    # Split for val and test
    shuffled = all_pairs[:]
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * 0.20)
    test_size = int(len(shuffled) * 0.20)

    val_pairs = shuffled[:val_size]
    test_pairs = shuffled[val_size:val_size + test_size]

    write_csv(VAL_CSV, val_pairs)
    write_csv(TEST_CSV, test_pairs)

    print(f"✅ Total Pairs: {len(all_pairs)}")
    print(f"📄 Train: {len(all_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")


if __name__ == '__main__':
    main()
