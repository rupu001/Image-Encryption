import numpy as np
import time, os
from typing import Optional

# def timeit(func):
#   def wrapper(*args, **kwargs):
#       start = time.perf_counter()
#       result = func(*args, **kwargs)
#       end = time.perf_counter()
#       elapsed = end - start
#       print(f'Time taken for {func.__name__}: {elapsed:.6f} seconds')
#       return result
#   return wrapper


class ZigZagGenerator:
    R_PLUS_C_DOWN = np.array([1, -1])
    R_MINUS_C_DOWN = np.array([1, 1])

    @classmethod
    def is_inside_grid(cls, x, y, n, m):
        return x >= 0 and y >= 0 and x < n and y < m

    @classmethod
    def load_from_existing(cls, existing_zigzag_path, p, a, n, m):
        path_name = f"{existing_zigzag_path}/{p}_{a}_{n}_{m}.npy"

        arr = None

        if os.path.exists(path_name):
            arr = np.load(path_name, allow_pickle=True)

        return arr

    @classmethod
    def save_to_existing(cls, existing_zigzag_path, p, a, n, m, arr):
        path_name = f"{existing_zigzag_path}/{p}_{a}_{n}_{m}.npy"
        if existing_zigzag_path:
            np.save(path_name, arr)

    @classmethod
    # @timeit
    def zigzag_p_a_wise(cls, p, a, n, m, existing_zigzag_path: Optional[str]):
        """
        Determines the Zigzag mode based on parameters p and a and
        Returns the corresponding sequence of 2d indices in their traverse order
        n and m are the matrix dimensions.
        """

        if existing_zigzag_path:
            arr = cls.load_from_existing(existing_zigzag_path, p, a, n, m)
            if arr is not None:
                return arr

        mode_index = (p - 1) * 4 + (a - 1) + 1
        # print(mode_index)

        if mode_index in (1, 7, 4, 5):
            generic_start_indices = [(i, 0) for i in range(n)][::-1] + [
                (0, j) for j in range(m)
            ][1::]
            index_list_list = [
                cls.generate_diagonal(index, "minus", n, m)
                for index in generic_start_indices
            ]
            if mode_index in (1, 4):
                to_reverse_indices = list(range(0, len(index_list_list), 2))
                for to_reverse_index in to_reverse_indices:
                    index_list_list[to_reverse_index] = index_list_list[
                        to_reverse_index
                    ][::-1]

                canonical_answer = np.array(sum(index_list_list, []))
                if mode_index == 4:
                    canonical_answer = canonical_answer[::-1]
                cls.save_to_existing(existing_zigzag_path, p, a, n, m, canonical_answer)

                return canonical_answer

            else:  # (7,5)
                to_reverse_indices = list(range(1, len(index_list_list), 2))
                for to_reverse_index in to_reverse_indices:
                    index_list_list[to_reverse_index] = index_list_list[
                        to_reverse_index
                    ][::-1]

                canonical_answer = np.array(sum(index_list_list, []))
                if mode_index == 5:
                    canonical_answer = canonical_answer[::-1]
                cls.save_to_existing(existing_zigzag_path, p, a, n, m, canonical_answer)

                return canonical_answer

        else:  # (2,6,3,8)
            generic_start_indices = [(0, j) for j in range(m)] + [
                (i, m - 1) for i in range(n)
            ][1::]

            index_list_list = [
                cls.generate_diagonal(index, "plus", n, m)
                for index in generic_start_indices
            ]

            if mode_index in (3, 6):
                to_reverse_indices = list(range(0, len(index_list_list), 2))
                for to_reverse_index in to_reverse_indices:
                    index_list_list[to_reverse_index] = index_list_list[
                        to_reverse_index
                    ][::-1]

                canonical_answer = np.array(sum(index_list_list, []))

                if mode_index == 6:
                    canonical_answer = canonical_answer[::-1]
                cls.save_to_existing(existing_zigzag_path, p, a, n, m, canonical_answer)

                return canonical_answer

            else:  # (2,8)
                to_reverse_indices = list(range(1, len(index_list_list), 2))
                for to_reverse_index in to_reverse_indices:
                    index_list_list[to_reverse_index] = index_list_list[
                        to_reverse_index
                    ][::-1]

                canonical_answer = np.array(sum(index_list_list, []))
                if mode_index == 8:
                    canonical_answer = canonical_answer[::-1]
                cls.save_to_existing(existing_zigzag_path, p, a, n, m, canonical_answer)

                return canonical_answer

    @classmethod
    def generate_diagonal(cls, start_pos: tuple, mode: str, n, m):
        if mode == "minus":
            delta = cls.R_MINUS_C_DOWN
        else:
            delta = cls.R_PLUS_C_DOWN

        diagonal_indices = []
        cur_pos = np.array(start_pos)
        while cls.is_inside_grid(*cur_pos, n, m):
            diagonal_indices.append(cur_pos.copy())
            cur_pos += delta

        return diagonal_indices
