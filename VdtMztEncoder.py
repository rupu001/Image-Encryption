import numpy as np
from zigzag import ZigZagGenerator
from vdt_utils import VDT_utils, timeit
from typing import Optional
import json, os
import pywt
from Crypto.Cipher import AES
from Crypto import Random


class VdtMztEncoder:

    @classmethod
    def apply_diffusion(
        cls, array: np.ndarray, index_array: np.ndarray, modulus_val: int
    ):
        """
        step by step apply the diffusion rather than using the boiled down formula. Input arrays are supposed to be one-dimensional.
        """

        out_array = np.zeros_like(array)
        for i in range(len(out_array)):
            out_array[i] = array[i] + index_array[i]
            if i > 0:
                out_array[i] += out_array[i - 1]
            out_array[i] %= modulus_val

        out_array = np.mod(out_array, modulus_val)
        return out_array

    @classmethod
    # @timeit
    def compute_plaintext_diffusion(
        cls, mat: np.ndarray, chaotic_x: float, chaotic_a: float
    ):
        """
        computes and returns a chatoic diffusion on the input matrix: mat.
        The shape can be (n,m)
        """

        # step 1:
        flattened_array = np.ndarray.flatten(mat)
        chaotic_sequence = VDT_utils.enhanced_logistic_map(
            chaotic_x, chaotic_a, flattened_array.shape[0]
        )
        index_array = np.argsort(chaotic_sequence)

        # step 2:
        once_diffused_array = cls.apply_diffusion(flattened_array, index_array, 256)

        # step 3:
        twice_diffused_array = cls.apply_diffusion(
            once_diffused_array[::-1], index_array[::-1], 256 #reverse of all the arrays
        )[
            ::-1
        ]  # check the orientation correctly
        output_matrix = twice_diffused_array.reshape(mat.shape)

        return output_matrix

    @classmethod
    # @timeit
    def permutate_cross_subband(
        cls,
        matrix: np.ndarray,
        r0: float,
        x0: float,
        r_array: np.ndarray,
        x_array: np.ndarray,
        existing_zigzag_path: Optional[str],
    ):
        """
        Input is supposed to be a 3d np.ndarray where the THIRD dimension is the channel dimension.
        """
        assert (
            np.ndarray.flatten(r_array).shape[0] == matrix.shape[2]
        ), "supply the same number of r_i^(1) [for i>0] values as the number of matrices"
        assert (
            np.ndarray.flatten(x_array).shape[0] == matrix.shape[2]
        ), "supply the same number of x_i^(1) [for i>0] values as the number of matrices"

        # step 1,2:
        q0 = VDT_utils.enhanced_logistic_map(
            VDT_utils.calculate_y_from_x(matrix, x0), r0, np.prod(matrix.shape)#Q0
        )

        # step 3:
        q0_3d = q0.reshape(matrix.shape)
        A = np.argsort(q0_3d, axis=-1)
        ## here P is the same thing as matrix
        P_prime = np.zeros_like(matrix)

        ## slower implementation but easier to understand
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                for k in range(matrix.shape[2]):
                    P_prime[i, j, k] = matrix[i, j, A[i, j, k]]

        # step 4:
        y_array = np.zeros_like(r_array)
        output_matrix = np.zeros_like(matrix)
        for i in range(y_array.shape[0]):
            y_array[i] = VDT_utils.calculate_y_from_x(P_prime[:, :, i], x_array[i])
            output_matrix[:, :, i] = cls.modified_zigzag_permutation(
                P_prime[:, :, i], y_array[i], r_array[i], existing_zigzag_path
            )

        return output_matrix

    @classmethod
    # @timeit
    def modified_zigzag_permutation(
        cls,
        matrix: np.ndarray,
        x0: float,
        a: float,
        existing_zigzag_path: Optional[str],
    ):
        """
        Performs the Modified Zigzag Transformation on the input matrix.
        """
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[0]  # Assuming a square matrix

        coords1, coords2 = VDT_utils.get_c1_c2_from_chaos(
            N, x0, a, existing_zigzag_path
        )

        # print("coords1: ",coords1+1)
        # print("coords2: ",coords2+1)

        # Swap the elements based on the recorded coordinates
        permuted_matrix = matrix.copy()

        # # shorter impl->
        # permuted_matrix[coords2[:,0],coords2[:,1]] =   permuted_matrix[coords1[:,0],coords1[:,1]]

        for (i1, j1), (i2, j2) in zip(coords1, coords2):
            permuted_matrix[i2, j2] = matrix[i1, j1]

        return permuted_matrix

    @classmethod
    # @timeit
    def encode_one_round(
        cls,
        image: np.ndarray,
        x_i_arr,
        r_i_arr,
        transformation_type: str,
        existing_zigzag_path: Optional[str],
        extra_secret_key: dict,
    ):
        assert image.ndim == 2
        assert image.shape[0] == image.shape[1], f"shape found: {image.shape}"
        assert image.shape[0] % 2 == 0

        if transformation_type == "vdt":
            LL, HD, VD, DD = VDT_utils.vdt(image)
        elif transformation_type == "dwt":
            LL, (HD, VD, DD) = pywt.dwt2(image, "haar")
        elif transformation_type == "dct":
            LL, HD, VD, DD = VDT_utils.create_4_components_by_dct(image)
        elif transformation_type == "cumsum2d":
            LL, HD, VD, DD = VDT_utils.create_4_components_by_cumsum2d(image)
        elif transformation_type == "AES":
            LL, HD, VD, DD = VDT_utils.create_4_components_by_AES(
                image,
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_AES":
            LL, HD, VD, DD = VDT_utils.create_4_components_by_AES(
                VDT_utils.cumsummation_2d(image),
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_VDT":
            LL, HD, VD, DD = VDT_utils.vdt(VDT_utils.cumsummation_2d(image))
        else:
            raise ValueError(
                f"No such implemented transformation as {transformation_type}"
            )

        N = HD.shape[0]

        HD_VD_DD_joined = np.concatenate(
            [HD.reshape(N, N, -1), VD.reshape(N, N, -1), DD.reshape(N, N, -1)], axis=2
        )

        hd_vd_dd_new = cls.permutate_cross_subband(
            HD_VD_DD_joined,
            r_i_arr[0],
            x_i_arr[0],
            r_i_arr[1:4],
            x_i_arr[1:4],
            existing_zigzag_path,
        )

        HD_new = hd_vd_dd_new[:, :, 0]
        VD_new = hd_vd_dd_new[:, :, 1]
        DD_new = hd_vd_dd_new[:, :, 2]

        plaintext_chaotic_x = VDT_utils.calculate_y_from_x(HD_VD_DD_joined, x_i_arr[0])
        # print("encode_side_LL:",LL)
        # print("x_i_arr[0]=",x_i_arr[0],"plaintext_chaotic_x=",plaintext_chaotic_x,"chaotic_a=",r_i_arr[0])
        LL_new = cls.compute_plaintext_diffusion(
            LL, chaotic_x=plaintext_chaotic_x, #y0 
            chaotic_a=r_i_arr[0]  #r0
        )

        # print("encode_side:",np.concatenate(
        #   [np.concatenate([LL_new,HD_new],axis=1),
        #   np.concatenate([VD_new,DD_new],axis=1)],axis=0
        # ))

        if transformation_type == "vdt":
            image_new = VDT_utils.ivdt(LL_new, HD_new, VD_new, DD_new)
        elif transformation_type == "dwt":
            image_new = pywt.idwt2([LL_new, [HD_new, VD_new, DD_new]], "haar")
        elif transformation_type == "dct":
            image_new = VDT_utils.join_4_components_by_idct(
                [LL_new, HD_new, VD_new, DD_new]
            )
        elif transformation_type == "cumsum2d":
            image_new = VDT_utils.join_4_components_by_icumsum2d(
                [LL_new, HD_new, VD_new, DD_new]
            )
        elif transformation_type == "AES":
            image_new = VDT_utils.join_4_components_by_iAES(
                [LL_new, HD_new, VD_new, DD_new],
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_AES":
            image_new = VDT_utils.inverse_cumsummation_2d(
                VDT_utils.join_4_components_by_iAES(
                    [LL_new, HD_new, VD_new, DD_new],
                    extra_secret_key["password"],
                    extra_secret_key["salt"],
                    extra_secret_key["iv"],
                )
            )
        elif transformation_type == "cumsum2d_VDT":
            image_new = VDT_utils.inverse_cumsummation_2d(
                VDT_utils.ivdt(LL_new, HD_new, VD_new, DD_new)
            )
        else:
            raise ValueError(
                f"No such implemented transformation as {transformation_type}"
            )

       

        image_new_2 = cls.modified_zigzag_permutation(
            image_new,
            x0=VDT_utils.calculate_y_from_x(image_new, x=x_i_arr[4]),
            a=r_i_arr[4],
            existing_zigzag_path=existing_zigzag_path,
        )

        if extra_secret_key.get("xor_of_chaos", False):
            image_new_2 = VDT_utils.xor_of_chaos(image_new_2, x_i_arr[4], r_i_arr[4])

        return image_new_2.copy()

    @classmethod
    @timeit
    def encode(
        cls,
        image: np.ndarray,
        secret_key: tuple,
        transformation_type="vdt",
        existing_zigzag_path: Optional[str] = None,
    ):
        """
        secret_key = (
          x0_1 (32-bit float) [0..1]
          x0_2 (32-bit float) [0..1]
          r0_1 (7-bit int + 25-bit float) [0..128)
          r0_2 (7-bit int + 25-bit float) [0..128)
          a1 (8-bit integers) [a11,a12,a13,a14] aij -> [0..127]
          a2 (8-bit integers) [a21,a22,a23,a24] aij -> [0..127]
          t0_1 (same format as r0_1)
          t0_2 (same format as r0_2)
          ...
        )
        """

        image_encoded_once, image_encoded_twice = cls.return_both_ciphers(
            image, secret_key, transformation_type, existing_zigzag_path
        )

        return image_encoded_twice

    # @classmethod
    # def encode_layerwise_one_round(
    #     cls,
    #     image: np.ndarray,
    #     x_i_arr,
    #     r_i_arr,
    #     transformation_type: str,
    #     existing_zigzag_path: Optional[str],
    #     extra_secret_key: dict,
    # ):
    #     if image.ndim == 2:
    #         return cls.encode_one_round(
    #             image,
    #             x_i_arr,
    #             r_i_arr,
    #             transformation_type,
    #             existing_zigzag_path,
    #             extra_secret_key,
    #         )
    #     elif image.ndim == 3:
    #         channel_wise_images = [
    #             cls.encode_one_round(
    #                 image[:, :, i],
    #                 x_i_arr,
    #                 r_i_arr,
    #                 transformation_type,
    #                 existing_zigzag_path,
    #                 extra_secret_key,
    #             )
    #             for i in range(image.shape[2])
    #         ]

    #         return np.stack(channel_wise_images, axis=-1)
    #     else:
    #         raise NotImplementedError(f"{image.ndim}-dim not implemented.")

    @classmethod
    def return_both_ciphers(
        cls,
        image: np.ndarray,
        secret_key: tuple,
        transformation_type: str,
        existing_zigzag_path: Optional[str],
    ):
        (x1, r1), (x2, r2) = VDT_utils.key_schedule(secret_key[:-1])
        extra_key1 = {}
        extra_key2 = {}

        for extra_key_fragment in ["xor_of_chaos", "password", "salt", "iv"]:
            if extra_key_fragment not in secret_key[-1].keys():
                continue

            extra_key1[extra_key_fragment] = secret_key[-1][extra_key_fragment][0]
            extra_key2[extra_key_fragment] = secret_key[-1][extra_key_fragment][1]

        image_encoded_once = cls.encode_one_round(
            image, x1, r1, transformation_type, existing_zigzag_path, extra_key1
        )
        image_encoded_twice = cls.encode_one_round(
            image_encoded_once,
            x2,
            r2,
            transformation_type,
            existing_zigzag_path,
            extra_key2,
        )

        return image_encoded_once, image_encoded_twice


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # img = cv2.imread('./data/inputs/Cover_gray.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("./data/inputs/cover_image1.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("./data/inputs/black_image.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(
    #     cv2.imread("./data/inputs/shreya.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    # )
    img = np.array(img, dtype="int32")

    plt.imshow(img, cmap='gray')
    plt.title("Plain Image")
    plt.show()

    transformation_type = "cumsum2d_VDT"
    per_round_xor=True
    last_round_xor=False

    extra_fragment_dict = {}
    if transformation_type in ["cumsum2d_AES", "AES"]:
        extra_fragment_dict["password"] = [
            "".join(map(chr, np.random.randint(33, 128, 100))),
            "".join(map(chr, np.random.randint(33, 128, 100))),
        ]
        extra_fragment_dict["salt"] = [
            list(os.urandom(AES.block_size)),
            list(os.urandom(AES.block_size)),
        ]
        extra_fragment_dict["iv"] = [
            list(Random.new().read(AES.block_size)),
            list(Random.new().read(AES.block_size)),
        ]
    
    if per_round_xor:
        extra_fragment_dict['xor_of_chaos'] = [
            True,True
        ]
    elif last_round_xor:
        extra_fragment_dict['xor_of_chaos'] = [
            False,True
        ]
    else :
        extra_fragment_dict['xor_of_chaos'] = [
            False,False
        ]

    secret_key = (
        np.random.random(),  # x0_1 (floating point between 0 and 1)
        np.random.random(),  # x0_2
        np.random.random() * 128,  # r0_1
        np.random.random() * 128,  # r0_2
        list(map(int, np.random.randint(0, 128, (4,)))),  # a1 # map converts numpy int to python int as json cannot directly store numpy array
        list(map(int, np.random.randint(0, 128, (4,)))),  # a2
        np.random.random() * 128,  # t0_1
        np.random.random() * 128,  # t0_2
        extra_fragment_dict,
    )

    img_encrypted = VdtMztEncoder.encode(
        img,
        secret_key,
        transformation_type=transformation_type,
        existing_zigzag_path="./data/zigzag",
    )

    plt.imshow(img_encrypted, cmap='gray')
    plt.title("Cipher Image")
    plt.show()

    np.save("data/outputs/cryptic_image.npy", img_encrypted)

    # print("saving image:",img_encrypted)

    with open("data/outputs/metadata.json", "w") as writer_f:
        json.dump({"secret_key": secret_key}, writer_f)

    # hd_vd_dd=np.random.randint(0,256,(8,8,3))
    # print(f"hd_vd_dd is: {hd_vd_dd}")
    # hd_vd_dd_permuted=VdtMztEncoder.permutate_cross_subband(hd_vd_dd, np.random.random(),np.random.random(),np.random.random(3),np.random.random((3)))
    # print(f"hd_vd_dd_permuted is: {hd_vd_dd_permuted}")

    # ll=np.random.randint(0,256,(4,4))
    # print(f"ll is {ll}")
    # ll_diffused=VdtMztEncoder.compute_plaintext_diffusion(ll,np.random.random(),np.random.random())
    # print(f"ll_diffused is {ll_diffused}")

    # # Create a sample 3x3 matrix for testing
    # matrix = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ])

    # # Perform Modified Zigzag Transformation
    # x0 = 0.5  # Initial value for the chaotic map
    # a = 3.8   # Control parameter for the logistic map
    # transformed_matrix = VdtMztEncoder.modified_zigzag_permutation(matrix, x0, a)

    # print(transformed_matrix)

    # # Example secret key
    # secret_key = (
    #     0.123456789,  # x0_1 (32-bit float)
    #     0.987654321,  # x0_2 (32-bit float)
    #     5.123456789,  # r0_1 (7-bit int + 25-bit float)
    #     7.987654321,  # r0_2 (7-bit int + 25-bit float)
    #     [12, 34, 56, 78],  # a1 (8-bit integers)
    #     [90, 23, 45, 67],  # a2 (8-bit integers)
    #     3.141592653,  # t0_1 (same format as r0_1)
    #     2.718281828   # t0_2 (same format as r0_2)
    # )

    # # Apply key scheduling
    # (group1, group2) = VdtMztEncoder.key_schedule(secret_key)

    # # Display the results
    # print("Group 1: Initial States and Control Parameters")
    # print("x1:", group1[0])
    # print("r1:", group1[1])

    # print("\nGroup 2: Initial States and Control Parameters")
    # print("x2:", group2[0])
    # print("r2:", group2[1])
