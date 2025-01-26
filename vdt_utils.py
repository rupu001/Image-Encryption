import numpy as np
from zigzag import ZigZagGenerator
import itertools, time
from typing import Optional
from scipy.fftpack import dct, idct
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto import Random
import os
from typing import Union


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"Time taken for {func.__name__}: {elapsed:.6f} seconds")
        return result

    return wrapper


# AES 256 encryption/decryption using pycrypto library


class AES_Encryption_Decryption:

    # pad with empty bytes at the end of the text
    # beacuse AES needs 16 byte blocks
    @classmethod
    def pad(cls, s: Union[bytes, bytearray]) -> tuple[bytes, int]:
        s = bytearray(s)
        block_size = 16
        remainder = len(s) % block_size
        padding_needed = block_size - remainder
        return bytes(s + bytes(padding_needed)), padding_needed

    # remove the extra bytes at the end
    @classmethod
    def unpad(cls, s: Union[bytes, bytearray], padding_needed: int) -> bytes:
        s = bytearray(s)
        if padding_needed > 0:
            s = s[:-padding_needed]

        return bytes(s)

    @classmethod
    def encrypt(
        cls,
        plain_bytes: list[int],
        password: str,
        salt: Union[list, bytes],
        iv: Union[list, bytes],
    ):
        if isinstance(salt, list):
            salt = bytes(salt)
        if isinstance(iv, list):
            iv = bytes(iv)

        # use the Scrypt KDF to get a private key from the password
        private_key = hashlib.scrypt(
            password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32
        )

        # pad text with spaces to be valid for AES CBC mode
        # padded_text = cls.pad(plain_bytes)
        padded_text = plain_bytes

        # create cipher config
        cipher_config = AES.new(private_key, AES.MODE_CBC, iv)

        return cipher_config.encrypt((bytes(padded_text)))

    @classmethod
    def decrypt(
        cls, cipher_text, password, salt: Union[list, bytes], iv: Union[list, bytes]
    ) -> list[int]:
        if isinstance(salt, list):
            salt = bytes(salt)
        if isinstance(iv, list):
            iv = bytes(iv)

            # generate the private key from the password and salt
            private_key = hashlib.scrypt(
                password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32
            )

            # create the cipher config
            cipher = AES.new(private_key, AES.MODE_CBC, iv)

            # decrypt the cipher text
            decrypted = cipher.decrypt(cipher_text)

            # unpad the text to remove the added spaces
            # original = cls.unpad(decrypted)
            original = list(decrypted)

            return original


class VDT_utils:

    @classmethod
    def enhanced_logistic_map(cls, x0, a, length):
        """
        Generates a chaotic sequence using the enhanced logistic map.
        """
        sequence = np.zeros(length)
        x = x0
        for i in range(length):
            x = np.sin(np.pi * a * x * (1 - x))
            sequence[i] = x
        return sequence

    @classmethod
    def log2_mod1(cls, x):
        """Helper function to compute (log2(x) mod 1)."""
        return np.log2(x) % 1

    @classmethod
    def key_schedule(cls, secret_key):
        """
        Generate original initial states {x(i)_j, r(i)_j} (i = 1,2, j = 0...4)
        using the secret key components.
        """
        # Unpack the secret key components
        (x0_1, x0_2, r0_1, r0_2, a1, a2, t0_1, t0_2) = secret_key

        # adding the dummy element to match indices as per the formula
        a1 = np.concatenate([[0], a1])
        a2 = np.concatenate([[0], a2])

        # Initialize arrays to store the initial states and control parameters
        x1, x2 = np.zeros(5), np.zeros(5)
        r1, r2 = np.zeros(5), np.zeros(5)

        # Assign initial values
        x1[0], x2[0] = x0_1, x0_2
        r1[0], r2[0] = r0_1, r0_2

        # Generate initial states and control parameters for j = 1, 2, 3, 4
        for j in range(1, 5):
            # For the first group (i = 1)
            x1[j] = (cls.log2_mod1(a1[j] + j) * x1[j - 1] + t0_1) % 1
            r1[j] = a1[j] + r1[j - 1] + t0_1

            # For the second group (i = 2)
            x2[j] = (cls.log2_mod1(a2[j] + j) * x2[j - 1] + t0_2) % 1
            r2[j] = a2[j] + r2[j - 1] + t0_2

        return (x1, r1), (x2, r2)
    # where all 4 are array of 5 values

    @classmethod
    # @timeit
    def generate_chaotic_indices(cls, N, x0, a):
        """
        Generates four chaotic sequences of length N and sorts them to get row and column indices.
        """

        full_chaotic_array = cls.enhanced_logistic_map(x0, a, 4 * N)
        R1 = full_chaotic_array[0:N]
        R2 = full_chaotic_array[N : 2 * N]
        C1 = full_chaotic_array[2 * N : 3 * N]
        C2 = full_chaotic_array[3 * N : 4 * N]

        # Sort the chaotic sequences to obtain permutation indices
        R1_indices = np.argsort(R1)
        R2_indices = np.argsort(R2)
        C1_indices = np.argsort(C1)
        C2_indices = np.argsort(C2)

        return R1_indices, R2_indices, C1_indices, C2_indices

    @classmethod
    def get_c1_c2_from_chaos(cls, N, x0, a, existing_zigzag_path: Optional[str]):

        # Step 1: Generate four chaotic sequences and corresponding indices
        R1, R2, C1, C2 = cls.generate_chaotic_indices(N, x0, a)

        # # @TESTING
        # R1=np.array([2,3,1]) -1
        # R2=np.array([2,1,3]) -1
        # C1=np.array([2,1,3]) -1
        # C2=np.array([1,2,3]) -1

        # Step 2: Calculate p and a parameters to select modes and starting points
        p1 = ((R1[0] + 1) % 2) + 1
        p2 = ((R2[0] + 1) % 2) + 1
        a1 = ((C1[0] + 1) % 4) + 1
        a2 = ((C2[0] + 1) % 4) + 1

        # Mode selection
        mode1_indices = ZigZagGenerator.zigzag_p_a_wise(
            p1, a1, N, N, existing_zigzag_path
        )
        mode2_indices = ZigZagGenerator.zigzag_p_a_wise(
            p2, a2, N, N, existing_zigzag_path
        )

        # print("mode1_indices: ",mode1_indices)
        # print("mode2_indices: ",mode2_indices)

        # Determine the starting points for scanning
        x1, y1 = R1[1], C1[-1]
        x2, y2 = R2[1], C2[-1]
        start_index_1 = np.array(
            np.where(np.all(mode1_indices == (x1, y1), axis=1))
        ).item()
        start_index_2 = np.array(
            np.where(np.all(mode2_indices == (x2, y2), axis=1))
        ).item()

        # print("start_index_1: ",start_index_1)
        # print("start_index_2: ",start_index_2)

        ordered_mode_indices_1 = np.roll(mode1_indices, -start_index_1, axis=0)
        ordered_mode_indices_2 = np.roll(mode2_indices, -start_index_2, axis=0)

        O1 = np.array(list(itertools.product(C1, R1))).reshape((N, N, 2))
        O2 = np.array(list(itertools.product(C2, R2))).reshape((N, N, 2))

        # print("O1: ",O1+1)
        # print("O2: ",O2+1)

        # print("ordered_mode_indices_1: ",ordered_mode_indices_1)
        # print("ordered_mode_indices_2: ",ordered_mode_indices_2)

        # Step 3: Perform Zigzag scanning on both matrices O1 and O2
        # Record the coordinates for swapping
        coords1 = O1[ordered_mode_indices_1[:, 0], ordered_mode_indices_1[:, 1]]
        coords2 = O2[ordered_mode_indices_2[:, 0], ordered_mode_indices_2[:, 1]]

        return coords1, coords2

    @classmethod
    def calculate_y_from_x(cls, arr: np.ndarray, x: float):
        avg = np.mean(arr)

        return avg - np.floor(avg) + x
    
    @classmethod
    def xor_of_chaos(cls, arr:np.ndarray, x:float, a:float, skip_n:int=10_000):
      assert skip_n>=0, f"skip_n has to be >=0. Supplied {skip_n}."
      random_matrix=np.array(cls.enhanced_logistic_map(x,a,arr.size+skip_n)[skip_n:]).reshape(arr.shape)
      random_matrix=np.mod(random_matrix*1e6,256)
      random_matrix=random_matrix.astype(arr.dtype)

      return np.bitwise_xor(arr,random_matrix)
    
    @classmethod
    def vdt(cls, image):
        # Get image dimensions (M x N) and ensure they are divisible by 2
        M, N = image.shape
        assert M % 2 == 0 and N % 2 == 0, "Image dimensions must be divisible by 2."

        # Initialize subbands: LL (original), HD, VD, DD (difference subbands)
        LL = np.zeros((M // 2, N // 2), dtype=np.int32)
        HD = np.zeros((M // 2, N // 2), dtype=np.int32)
        VD = np.zeros((M // 2, N // 2), dtype=np.int32)
        DD = np.zeros((M // 2, N // 2), dtype=np.int32)

        # Step 1: Divide the image into 2x2 blocks
        for i in range(0, M, 2):
            for j in range(0, N, 2):
                # Extract 2x2 block
                block = image[i : i + 2, j : j + 2]

                # Step 2: Calculate differences between the first pixel (1,1) with others
                LL[i // 2, j // 2] = block[0, 0]
                HD[i // 2, j // 2] = block[0, 1] - block[0, 0]
                VD[i // 2, j // 2] = block[1, 0] - block[0, 0]
                DD[i // 2, j // 2] = block[1, 1] - block[0, 0]

        # Step 4: Perform modulo 256 operation on all subbands
        LL = np.mod(LL, 256)
        HD = np.mod(HD, 256)
        VD = np.mod(VD, 256)
        DD = np.mod(DD, 256)

        return LL, HD, VD, DD

    @classmethod
    def ivdt(cls, LL, HD, VD, DD):
        # Get the size of the input matrices
        N = LL.shape[0]

        # Initialize a list to store the 2x2 matrices
        two_rows_list = []

        # Iterate over each index (i, j) in the N x N matrices
        for i in range(N):
            two_by_two_list = []
            for j in range(N):
                # Create a 2x2 matrix from the current index in each matrix
                matrix_2x2 = np.array([[LL[i, j], HD[i, j]], [VD[i, j], DD[i, j]]])

                matrix_2x2[[0, 1, 1], [1, 0, 1]] += matrix_2x2[0, 0]
                matrix_2x2 %= 256

                two_by_two_list.append(matrix_2x2)
            two_rows_list.append(np.concatenate(two_by_two_list, axis=1))

        full_image = np.concatenate(two_rows_list, axis=0)

        return full_image

    @classmethod
    def apply_dct(cls, image):
        """
        Apply 2D DCT on an image.
        """
        return dct(dct(image.T, norm="ortho").T, norm="ortho")

    @classmethod
    def apply_idct(cls, dct_image):
        """
        Apply 2D IDCT to reconstruct the image.
        """
        return idct(idct(dct_image.T, norm="ortho").T, norm="ortho")

    @classmethod
    def divide_into_4_components(cls, matrix: np.ndarray):
        assert np.ndim(matrix) == 2, "2D image required"
        assert (
            matrix.shape[0] % 2 == 0 and matrix.shape[1] % 2 == 0
        ), "Even shaped dimensions required"

        n, m = matrix.shape

        return (
            matrix[0 : n // 2, 0 : m // 2].copy(),
            matrix[0 : n // 2, m // 2 :].copy(),
            matrix[n // 2 :, 0 : m // 2].copy(),
            matrix[n // 2 :, m // 2 :].copy(),
        )

    @classmethod
    def join_4_components(cls, array_of_matrices):
        assert len(array_of_matrices) == 4, "4 components needed"

        joined_matrix = np.concatenate(
            [
                np.concatenate(array_of_matrices[:2], axis=1),
                np.concatenate(array_of_matrices[2:], axis=1),
            ],
            axis=0,
        )

        return joined_matrix

    @classmethod
    def create_4_components_by_dct(cls, image: np.ndarray):
        decomposed_image = cls.apply_dct(image)
        return cls.divide_into_4_components(decomposed_image)

    @classmethod
    def join_4_components_by_idct(cls, array_of_components: list[np.ndarray]):
        joined_matrix = cls.join_4_components(array_of_components)
        return cls.apply_idct(joined_matrix)

    @classmethod
    def cumsummation_2d(cls, matrix: np.ndarray, modulus_val=256):
        """
        Beware of overflow situations in the cumsum stage
        """
        horizontally_summed_matrix = np.mod(np.cumsum(matrix, axis=1), modulus_val)
        bothways_summed_matrix = np.mod(
            np.cumsum(horizontally_summed_matrix, axis=0), modulus_val
        )

        return bothways_summed_matrix

    @classmethod
    def inverse_cumsummation_2d(cls, cumsummed_matrix: np.ndarray, modulus_val=256):
        vertically_diffed_matrix = np.mod(
            np.diff(cumsummed_matrix, n=1, axis=0, prepend=0) + modulus_val, modulus_val
        )
        original_matrix = np.mod(
            np.diff(vertically_diffed_matrix, n=1, axis=1, prepend=0) + modulus_val,
            modulus_val,
        )

        return original_matrix

    @classmethod
    def create_4_components_by_cumsum2d(cls, image):
        decomposed_image = cls.cumsummation_2d(image)
        return cls.divide_into_4_components(decomposed_image)

    @classmethod
    def join_4_components_by_icumsum2d(cls, array_of_components):
        joined_matrix = cls.join_4_components(array_of_components)
        return cls.inverse_cumsummation_2d(joined_matrix)

    @classmethod
    def create_4_components_by_AES(cls, image: np.ndarray, password, salt, iv):
        byte_image = bytes(list(image.flatten()))
        assert len(byte_image) % 16 == 0, "Number of pixels must be divisible by 16"
        #as block size is 16

        encrypted_bytes = AES_Encryption_Decryption.encrypt(
            byte_image, password, salt, iv
        )
        encrypted_image = np.array(list(encrypted_bytes), dtype=image.dtype).reshape(
            image.shape
        )

        return cls.divide_into_4_components(encrypted_image)

    @classmethod
    def join_4_components_by_iAES(cls, array_of_components, password, salt, iv):
        joined_matrix = cls.join_4_components(array_of_components)

        byte_image = bytes(list(joined_matrix.flatten()))
        assert len(byte_image) % 16 == 0, "Number of pixels must be divisible by 16"

        decrypted_bytes = AES_Encryption_Decryption.decrypt(
            byte_image, password, salt, iv
        )
        decrypted_image = np.array(
            list(decrypted_bytes), dtype=joined_matrix.dtype
        ).reshape(joined_matrix.shape)

        return decrypted_image


if __name__ == "__main__":
    pass

    # # Example 4x4 image
    # image = np.array([
    #     [115, 75, 10, 7],
    #     [200, 205, 20, 56],
    #     [53, 85, 36, 48],
    #     [50, 72, 92, 27]
    # ], dtype=np.int32)

    # # Apply VDT transformation
    # LL, HD, VD, DD = VDT_utils.vdt_transform(image)

    # # Example matrices (for demonstration)
    # LL = np.array([
    #     [115, 10],
    #     [53, 36]
    # ], dtype=np.int32)

    # HD = np.array([
    #     [216, 253],
    #     [32, 12]
    # ], dtype=np.int32)

    # VD = np.array([
    #     [85, 10],
    #     [253, 56]
    # ], dtype=np.int32)

    # DD = np.array([
    #     [90, 46],
    #     [19, 247]
    # ], dtype=np.int32)

    # # Generate the list of 2x2 matrices
    # result = VDT_utils.ivdt(LL, HD, VD, DD)

    # print("result:\n", result)

    ## @TEST vdt ivdt
    # image=np.random.randint(0,256,(128,128))
    # tup=VDT_utils.vdt(image)
    # image1=VDT_utils.ivdt(*tup)

    # assert(np.all(image1==image))
    # print("passed vdt ivdt!")

    ## @TEST dct idct
    # image = np.random.randint(0,256,(128,128))
    # list_of_components=VDT_utils.create_4_components_by_dct(image)
    # print(len(list_of_components))
    # recon_image=np.round(VDT_utils.join_4_components_by_idct(list_of_components))

    # assert np.all(image==recon_image)
    # print("passed dct idct!")

    # rand_stuff=(np.random.random((128,128))*2-1)*256
    # _img=VDT_utils.apply_idct(rand_stuff)
    # __rand_stuff=VDT_utils.apply_dct(_img)

    # print(np.all(np.isclose(rand_stuff,__rand_stuff)))

    # # @TEST cumsum2d icumsum2d
    # image=np.random.randint(0,256,(128,128))
    # cumsummed_image=VDT_utils.cumsummation_2d(image,256)
    # icumsummed_image=VDT_utils.inverse_cumsummation_2d(cumsummed_image,256)
    # print(np.all(image==icumsummed_image))

    ## @TEST AES_Enc_Dec
    # import tqdm
    # for _ in tqdm.tqdm(range(100)):

    #   img=np.random.randint(0,256,(128,128))
    #   password = ''.join(map(chr,np.random.randint(33,128,10)))
    #   # generate a random salt
    #   salt = os.urandom(AES.block_size)

    #   # generate a random iv
    #   iv = Random.new().read(AES.block_size)

    #   # First let us encrypt secret message
    #   encrypted = AES_Encryption_Decryption.encrypt(list(img.flatten()), password,salt,iv)

    #   # Let us decrypt using our original password
    #   decrypted_list = AES_Encryption_Decryption.decrypt(encrypted, password,salt,iv)
    #   decrypted_img = np.array(decrypted_list,dtype=img.dtype).reshape(img.shape)

    #   assert(np.all(img==decrypted_img))
    # print("AES passed!")

    # # @TEST join and divide AES
    # password = ''.join(map(chr,np.random.randint(33,128,10)))
    # salt = list(os.urandom(AES.block_size))
    # iv = list(Random.new().read(AES.block_size))

    # image=np.random.randint(0,256,(128,128))
    # list_of_components=VDT_utils.create_4_components_by_AES(image,password,salt,iv)
    # recon_image=np.round(VDT_utils.join_4_components_by_iAES(list_of_components,password,salt,iv))

    # assert(np.all(image==recon_image))

    # print("Passed join and divide AES!")
