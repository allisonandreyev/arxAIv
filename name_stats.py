from collections import Counter

def analyze_names(name_string):
    lines = [n.strip() for n in name_string.split("\n") if n.strip()]

    full_name_counter = Counter()
    first_name_counter = Counter()
    last_name_counter = Counter()

    for name in lines:
        full_name_counter[name] += 1

        parts = name.split()
        if len(parts) >= 1:
            first_name_counter[parts[0]] += 1
        if len(parts) >= 2:
            last_name_counter[parts[-1]] += 1

    print("\nMOST COMMON FULL NAMES:")
    for name, count in full_name_counter.most_common():
        print(f"{name} — {count}")

    print("\nMOST COMMON FIRST NAMES:")
    for name, count in first_name_counter.most_common():
        print(f"{name} — {count}")

    print("\nMOST COMMON LAST NAMES:")
    for name, count in last_name_counter.most_common():
        print(f"{name} — {count}")


# ---------------- EXAMPLE ----------------
if __name__ == "__main__":
    raw_names = ""
    analyze_names(raw_names)
