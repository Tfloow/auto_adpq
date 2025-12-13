import time

from tqdm import tqdm

with tqdm(
    total=10,
    desc="⏳ Loading Layer Weights",
    unit="layer"
) as pbar:
    for _ in range(10):
        time.sleep(0.1)  # Simulate work being done

        pbar.update(1)
        pbar.set_postfix(finished="layer_name", refresh=True)

    # pbar finalize
    pbar.close()
