import numpy as np
from scipy.stats.stats import pearsonr
from math import log10
from skimage.metrics import structural_similarity as ssim
from scipy.stats import norm
import seaborn as sn


class Performance:
    @classmethod
    def calculate_corr(cls, image: np.ndarray, how: str):
        assert image.ndim == 2, "Give a 2d image."

        X, Y = cls.__get_pixel_sequences(image, how)

        # only take the r value aka the coefficient of correlation
        return pearsonr(X, Y)[0]

    @classmethod
    def __get_pixel_sequences(cls, image: np.ndarray, how: str):
        n, m = image.shape

        if how.lower() == "vertical":
            return image[: n - 1, :].flatten().copy(), image[1:n, :].flatten().copy()
        elif how.lower() == "horizontal":
            return image[:, : m - 1].flatten().copy(), image[:, 1:m].flatten().copy()
        elif how.lower() == "diagonal":
            list1 = []
            list2 = []

            start_indices = [(i, 0) for i in range(n)] + [(0, j) for j in range(1, m)]

            for start_index in start_indices:
                cur_index = np.array(start_index)
                while np.all(cur_index < [n, m]):  ## within range
                    if np.prod(cur_index) != 0:  ## not at starting position
                        list2.append(image[cur_index[0], cur_index[1]])
                    if not np.any(
                        cur_index == [n - 1, m - 1]
                    ):  ## not at the ending position
                        list1.append(image[cur_index[0], cur_index[1]])

                    cur_index += 1

            return list1, list2
        else:
            raise NotImplemented()

    @classmethod
    def binarize_secret_key(cls, secret_key: list):
        assert len(secret_key) == 8, "Secret key invalid."

        ## index 4 and 5 are arrays of ints. Rest are floats
        byte_values = []
        for idx, elem in enumerate(secret_key):
            start_dtype = "float32"
            if idx == 4 or idx == 5:
                start_dtype = "int8"

            tmp_byte_values = (
                np.array([elem], dtype=start_dtype).view(dtype="uint8").flatten()
            )
            byte_values.extend(tmp_byte_values)

        return [
            int(bool_val)
            for bool_val in "".join(
                str.zfill(bin(byte_)[2::], 8) for byte_ in byte_values
            )
        ]

    @classmethod
    def create_secret_key_from_binary(cls, boolean_list: list):
        assert len(boolean_list) == 256, "Invalid binary repr of secret key."
        int8_list = [
            int("".join(map(str, boolean_list[i : i + 8])), base=2)
            for i in range(0, 256, 8)
        ]

        elem0 = (
            np.array(int8_list[0:4], dtype="uint8").view(dtype="float32").copy().item()
        )
        elem1 = (
            np.array(int8_list[4:8], dtype="uint8").view(dtype="float32").copy().item()
        )
        elem2 = (
            np.array(int8_list[8:12], dtype="uint8").view(dtype="float32").copy().item()
        )
        elem3 = (
            np.array(int8_list[12:16], dtype="uint8")
            .view(dtype="float32")
            .copy()
            .item()
        )
        elem4 = list(
            np.array(int8_list[16:20], dtype="uint8").view(dtype="int8").copy()
        )
        elem5 = list(
            np.array(boolean_list[20:24], dtype="uint8").view(dtype="int8").copy()
        )
        elem6 = (
            np.array(boolean_list[24:28], dtype="uint8")
            .view(dtype="float32")
            .copy()
            .item()
        )
        elem7 = (
            np.array(boolean_list[28:32], dtype="uint8")
            .view(dtype="float32")
            .copy()
            .item()
        )

        return [elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7]

    @classmethod
    def calculate_NBCR(cls, image1: np.ndarray, image2: np.ndarray):
        assert np.prod(image1.shape) == np.prod(image2.shape)

        binary1 = "".join(bin(pixel)[2::] for pixel in image1.flatten())
        binary2 = "".join(bin(pixel)[2::] for pixel in image2.flatten())

        return sum([val1 != val2 for val1, val2 in zip(binary1, binary2)]) / len(
            binary1
        )

    @classmethod
    def calculate_NPCR(cls, image1: np.ndarray, image2: np.ndarray):
        assert np.prod(image1.shape) == np.prod(image2.shape)

        return np.sum(image1 != image2) / np.prod(image1.shape)

    @classmethod
    def calculate_UACI(
        cls,
        image1: np.ndarray,
        image2: np.ndarray,
    ):
        assert np.prod(image1.shape) == np.prod(image2.shape)

        return (np.sum(np.abs(image1 - image2))) / (
            255 * np.prod(image1.shape)
        )  # differs from what the paper says

    @classmethod
    def calculate_v_alpha(cls, L, N, M, alpha):
        """
        L is the max value any pixel can take in the image.
        L is supposed to be 255 for an 8bit image.
        """
        phi_alpha_inv = norm.ppf(alpha)

        # Note: here phi_alpha_inv is a negative number for alpha<0.5
        # Putting only a 'plus' sign here matched with the results
        # of the paper.
        return (L + phi_alpha_inv * np.sqrt(L / (M * N))) / (L + 1)

    @classmethod
    def calculate_thetas(cls, L, N, M, l, alpha):
        """
        L is the max value any pixel can take in the image.
        L is supposed to be 255 for an 8bit image.
        """
        phi_inv = norm.ppf(alpha / 2)
        const_part = (L + 2) / (3 * L + 3)
        var_part = (
            phi_inv
            * (L + 2)
            * (L * L + 2 * L + 3)
            / (18 * np.square(l + 1) * L * M * N)
        )

        return (
            const_part - var_part,
            const_part + var_part,
        )

    @classmethod
    def calculate_psnr(cls, original_image, reconstructed_image):
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((original_image - reconstructed_image) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the signal.
            return float("inf")
        PIXEL_MAX = 255.0
        psnr_value = 20 * log10(PIXEL_MAX / np.sqrt(mse))
        return psnr_value

    @classmethod
    def calculate_ssim(cls, original_image, reconstructed_image):
        ssim_value, _ = ssim(original_image, reconstructed_image, full=True)
        return ssim_value


if __name__ == "__main__":

    from PIL import Image
    import matplotlib.pyplot as plt
    import json, random
    from VdtMztEncoder import VdtMztEncoder
    from VdtMztDecoder import VdtMztDecoder
    from attacks import add_gaussian_noise, apply_occlusion
    from tqdm import tqdm
    import cv2.cv2

    original_img = np.array(
        cv2.imread("data/inputs/cover_image1.png", cv2.IMREAD_GRAYSCALE)
        # cv2.imread("data/inputs/shreya.png", cv2.IMREAD_COLOR)
    )
    cryptic_img = np.array(
        np.load("data/outputs/cryptic_image.npy", allow_pickle=True),
        dtype="int32",
    )

    # if original_img.ndim==3:
    #   original_img=cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)

    original_img=np.array(original_img,dtype='int32')

    with open("./data/outputs/metadata.json", "r") as json_reader_f:
        secret_key = json.load(json_reader_f)["secret_key"]

    # # correlation coefficient calculation
    # corr_hows=["horizontal","vertical","diagonal"]
    # original_corr_values=list(map(lambda how:Performance.calculate_corr(original_img,how),corr_hows))
    # cryptic_corr_values=list(map(lambda how:Performance.calculate_corr(cryptic_img,how),corr_hows))

    # print("original_corr_values: ",original_corr_values)
    # print("cryptic_corr_values: ",cryptic_corr_values)

    # # NBCR values plotted
    # binarized_secret_key=Performance.binarize_secret_key(secret_key)
    # NBCR_values=[]
    # for idx in tqdm(range(len(binarized_secret_key))):
    #   # random_index=random.randint(0,len(binarized_secret_key)-1)
    #   binarized_secret_key[idx]=1-binarized_secret_key[idx] # flip the bit
    #   changed_secret_key=Performance.create_secret_key_from_binary(binarized_secret_key)
    #   changed_cryptic_img=VdtMztEncoder.encode(original_img,changed_secret_key,transformation_type='vdt',existing_zigzag_path='./data/zigzag')

    #   binarized_secret_key[idx]=1-binarized_secret_key[idx] # flip back the bit

    #   NBCR_values.append(Performance.calculate_NBCR(original_img,changed_cryptic_img))

    # plt.plot(NBCR_values)
    # plt.show()

    # # Historgram plotting for 1 bit-changed secret key
    # binarized_secret_key=Performance.binarize_secret_key(secret_key[:-1])
    # random_index=random.randint(0,len(binarized_secret_key)-1)
    # binarized_secret_key[random_index]=1-binarized_secret_key[random_index] # flip the bit
    # #print("secret key", secret_key[-1])
    # changed_secret_key=Performance.create_secret_key_from_binary(binarized_secret_key)+[secret_key[-1]]
    # changed_cryptic_img=VdtMztEncoder.encode(original_img,changed_secret_key,transformation_type='cumsum2d_VDT',existing_zigzag_path='./data/zigzag')
    # binarized_secret_key[random_index]=1-binarized_secret_key[random_index] # flip back the bit

    # plt.hist(original_img)
    # plt.title("original_img")
    # plt.show()
    # plt.hist(cryptic_img)
    # plt.title("cryptic_img")
    # plt.show()
    # plt.hist(changed_cryptic_img)
    # plt.title("changed_cryptic_img")
    # plt.show()
    # plt.hist(np.abs(cryptic_img-changed_cryptic_img))
    # plt.title("|cryptic_img - changed_cryptic_img|")
    # plt.show()

    # NPCR and UACI analysis
    cipher1, cipher2 = VdtMztEncoder.return_both_ciphers(
        original_img,
        secret_key,
        transformation_type="cumsum2d_VDT",
        existing_zigzag_path="./data/zigzag",
    )

    one_bit_changed_img = original_img.copy()
    # take a random cell in the image matrix. Take a random bit index and flip it.
    rand_choice_index = (0, 0)#np.random.randint(0, original_img.shape, original_img.ndim)
    random_8_bit_bitmask = 1 << np.random.randint(0, 8)
    one_bit_changed_img[
        tuple(rand_choice_index)
    ] ^= random_8_bit_bitmask
    print(
        f"rand_choice_index:{rand_choice_index}, random_8_bit_bitmask:{random_8_bit_bitmask}"
    )

    one_bit_changed_cipher = VdtMztEncoder.encode(
        one_bit_changed_img,
        secret_key,
        transformation_type="cumsum2d_VDT",
        existing_zigzag_path="./data/zigzag",
    )

    npcr1 = Performance.calculate_NPCR(original_img, cipher1)
    npcr2 = Performance.calculate_NPCR(cipher1, cipher2)
    npcr3 = Performance.calculate_NPCR(cipher2, one_bit_changed_cipher)
    v_alpha = Performance.calculate_v_alpha(255, *original_img.shape, alpha=0.05)

    uaci1 = Performance.calculate_UACI(original_img, cipher1)
    uaci2 = Performance.calculate_UACI(cipher1, cipher2)
    uaci3 = Performance.calculate_UACI(cipher2, one_bit_changed_cipher)

    # could not get the l value here.
    # theta_s=Performance.calculate_thetas(

    # )

    # print("npcr1: ", npcr1)
    # print("npcr2: ", npcr2)
    print("npcr: ", npcr3)
    # print("v_alpha: ", v_alpha)

    # print("uaci1: ", uaci1)
    # print("uaci2: ", uaci2)
    print("uaci: ", uaci3)

    # # PSNR analysis
    # decrypted_img=VdtMztDecoder.decode(cryptic_img,secret_key,transformation_type='vdt',existing_zigzag_path='./data/zigzag')

    # psnr_val=Performance.calculate_psnr(original_img,decrypted_img)
    # print("psnr value: ",psnr_val)

    # # SSIM analysis
    # decrypted_img=VdtMztDecoder.decode(cryptic_img,secret_key,transformation_type='vdt',existing_zigzag_path='./data/zigzag')
    # ssim_val=Performance.calculate_ssim(original_img,decrypted_img)
    # print("ssim value: ",ssim_val)

    # # Efficiency Analysis

    # # run a few times with existing_zigzag_path='./data/zigzag' to cache
    # # all the zigzag patterns

    # decrypted_img=VdtMztDecoder.decode(cryptic_img,secret_key,transformation_type='vdt',existing_zigzag_path='./data/zigzag')

    # # Occlusion attack

    # occl_pos=np.array(
    #   [100,100]
    # )
    # occl_dim=np.array(
    #   [1,1]
    # )

    # occluded_cryptic_img=apply_occlusion(cryptic_img,occlusion_size=occl_dim,position=occl_pos)
    # occluded_decrypted_img=VdtMztDecoder.decode(occluded_cryptic_img,secret_key)

    # plt.title("occluded decrypted image.")
    # plt.imshow(occluded_decrypted_img)
    # plt.show()

    # # Gaussian attack
    # gaussian_image=add_gaussian_noise(cryptic_img)
    # gaussian_decrypted_img=VdtMztDecoder.decode(gaussian_image,secret_key)

    # plt.title("gaussian_decrypted_img")
    # plt.imshow(gaussian_decrypted_img)
    # plt.show()
