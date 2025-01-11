import matplotlib.pyplot as plt

def generate_coords(spacing=1):
    # Left side rows (top to bottom)
    left_side_rows = [
        [12, 11],
        [10, 9, 8],
        [7, 6],
        [5, 4, 3],
        [2, 1]
    ]
    # Right side rows (top to bottom)
    right_side_rows = [
        [24, 23],
        [22, 21, 20],
        [19, 18],
        [17, 16, 15],
        [14, 13]
    ]
    coords_left = {}
    coords_right = {}
    for row_idx, row_data in enumerate(left_side_rows):
        # Calculate y for each row
        spacing_h = 0.5 * spacing
        y = -row_idx * spacing_h
        # Handle rows with 2 or 3 nodes
        if len(row_data) == 3:
            for col_idx, node_id in enumerate(row_data):
                x = col_idx * spacing
                coords_left[node_id] = (x, y)
        elif len(row_data) == 2:
            for col_idx, node_id in enumerate(row_data):
                x = col_idx * spacing + spacing * 0.5
                coords_left[node_id] = (x, y)
        else:
            raise ValueError("Each row must contain 2 or 3 nodes")

    for row_idx, row_data in enumerate(right_side_rows):
        # Same logic for right side
        y = -row_idx * spacing_h
        if len(row_data) == 3:
            for col_idx, node_id in enumerate(row_data):
                x = col_idx * spacing
                coords_right[node_id] = (x, y)
        elif len(row_data) == 2:
            for col_idx, node_id in enumerate(row_data):
                x = col_idx * spacing + spacing * 0.5
                coords_right[node_id] = (x, y)
        else:
            raise ValueError("Each row must contain 2 or 3 nodes")

    return coords_left, coords_right

def plot_coords(coords, title="EEG Left Side Mapping"):
    """Scatter plot of mapped nodes."""
    for label, (x, y) in coords.items():
        plt.scatter(x, y, c="blue")
        plt.text(x + 0.003, y + 0.003, str(label), fontsize=9)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("off")

def corresponding_pairs():
    pair_lTor = {
        12: 23,
        11: 24,
        10: 20,
        9: 21,
        8: 22,
        7: 18,
        6: 19,
        5: 15,
        4: 16,
        3: 17,
        2: 13,
        1: 14
    }
    pair_rTol = {v: k for k, v in pair_lTor.items()}
    pair_total = pair_lTor.copy()
    pair_total.update(pair_rTol)
    return pair_total, pair_lTor, pair_rTol

if __name__ == "__main__":
    left_side_coords, right_side_coords = generate_coords(spacing=0.1)
    print("Left side node-to-coord map:")
    for k, v in left_side_coords.items():
        print(f"Node {k} -> {v}")
    for k, v in right_side_coords.items():
        print(f"Node {k} -> {v}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_coords(left_side_coords, title="EEG Left Side Mapping")
    plt.subplot(1, 2, 2)
    plot_coords(right_side_coords, title="EEG Right Side Mapping")
    plt.show()
