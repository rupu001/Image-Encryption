import numpy as np
from zigzag import ZigZagGenerator
from vdt_utils import VDT_utils, timeit
import json
import matplotlib.pyplot as plt
from ctwavelet import LiftingScheme
from typing import Optional
import pywt


class VdtMztDecoder:

    @classmethod
    def unapply_diffusion(
        cls, diffused_array: np.ndarray, index_array: np.ndarray, modulus_val: int
    ):

        original_array = np.zeros_like(diffused_array)
        for i in range(original_array.shape[0]):
            if i == 0:
                original_array[i] = (
                    diffused_array[i] - index_array[i] + modulus_val
                ) % modulus_val
            else:
                original_array[i] = (
                    diffused_array[i]
                    - diffused_array[i - 1]
                    - index_array[i]
                    + 2 * modulus_val
                ) % modulus_val

        return original_array

    @classmethod
    def uncompute_plaintext_diffusion(
        cls, doubly_diffused_mat: np.ndarray, chaotic_x: float, chaotic_a: float
    ):
        doubly_diffused_flattened_array = doubly_diffused_mat.flatten()
        index_array = np.argsort(
            VDT_utils.enhanced_logistic_map(
                chaotic_x, chaotic_a, doubly_diffused_flattened_array.shape[0]
            )
        )

        # forward pass logic:
        # diff->inverse->diff->inverse
        # apply "un"process backwards
        singly_dediffused_array = cls.unapply_diffusion(
            doubly_diffused_flattened_array[::-1], index_array[::-1], 256
        )[::-1]
        original_array = cls.unapply_diffusion(
            singly_dediffused_array, index_array, 256
        )

        return original_array.reshape(doubly_diffused_mat.shape)

    @classmethod
    def unpermutate_cross_subband(
        cls,
        permuted_matrix: np.ndarray,
        r0: float,
        x0: float,
        r_array: np.ndarray,
        x_array: np.ndarray,
        existing_zigzag_path: Optional[str],
    ):
        q0 = VDT_utils.enhanced_logistic_map(
            VDT_utils.calculate_y_from_x(permuted_matrix, x0),
            r0,
            np.prod(permuted_matrix.shape),
        )
        q0_3d = q0.reshape(permuted_matrix.shape)
        A = np.argsort(q0_3d, axis=-1)

        y_array = np.zeros_like(r_array)
        bandwise_unzigzagged_mat = np.zeros_like(permuted_matrix)
        for i in range(x_array.shape[0]):
            y_array[i] = VDT_utils.calculate_y_from_x(
                permuted_matrix[:, :, i], x_array[i]
            )
            bandwise_unzigzagged_mat[:, :, i] = cls.unmodified_zigzag_permutation(
                permuted_matrix[:, :, i], y_array[i], r_array[i], existing_zigzag_path
            )

        fully_unzigzagged_mat = np.zeros_like(permuted_matrix)
        for i in range(permuted_matrix.shape[0]):
            for j in range(permuted_matrix.shape[1]):
                for k in range(permuted_matrix.shape[2]):
                    fully_unzigzagged_mat[i, j, A[i, j, k]] = bandwise_unzigzagged_mat[
                        i, j, k
                    ]

        return fully_unzigzagged_mat

    @classmethod
    def unmodified_zigzag_permutation(
        cls,
        permuted_matrix: np.ndarray,
        x0: float,
        a: float,
        existing_zigzag_path: Optional[str],
    ):
        N = permuted_matrix.shape[0]
        coords1, coords2 = VDT_utils.get_c1_c2_from_chaos(
            N, x0, a, existing_zigzag_path
        )

        matrix = np.zeros_like(permuted_matrix)

        for (i1, j1), (i2, j2) in zip(coords1, coords2):
            matrix[i1, j1] = permuted_matrix[i2, j2]

        return matrix

    @classmethod
    def decode_one_round(
        cls,
        cipher_image: np.ndarray,
        x_i_arr,
        r_i_arr,
        transformation_type: str,
        existing_zigzag_path: Optional[str],
        extra_secret_key: dict,
    ):
        if extra_secret_key.get("xor_of_chaos", False):
            cipher_image = VDT_utils.xor_of_chaos(cipher_image, x_i_arr[4], r_i_arr[4])

        unglobal_permuted_image = cls.unmodified_zigzag_permutation(
            cipher_image,
            x0=VDT_utils.calculate_y_from_x(cipher_image, x=x_i_arr[4]),
            a=r_i_arr[4],
            existing_zigzag_path=existing_zigzag_path,
        )

        if transformation_type == "vdt":
            LL_, HD_, VD_, DD_ = VDT_utils.vdt(unglobal_permuted_image)
        elif transformation_type == "dwt":
            LL_, (HD_, VD_, DD_) = pywt.dwt2(unglobal_permuted_image, "haar")
        elif transformation_type == "dct":
            LL_, HD_, VD_, DD_ = VDT_utils.create_4_components_by_dct(
                unglobal_permuted_image
            )
        elif transformation_type == "cumsum2d":
            LL_, HD_, VD_, DD_ = VDT_utils.create_4_components_by_cumsum2d(
                unglobal_permuted_image
            )
        elif transformation_type == "AES":
            LL_, HD_, VD_, DD_ = VDT_utils.create_4_components_by_AES(
                unglobal_permuted_image,
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_AES":
            LL_, HD_, VD_, DD_ = VDT_utils.create_4_components_by_AES(
                VDT_utils.cumsummation_2d(unglobal_permuted_image),
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_VDT":
            LL_, HD_, VD_, DD_ = VDT_utils.vdt(
                VDT_utils.cumsummation_2d(unglobal_permuted_image)
            )
        else:
            raise ValueError(
                f"No such implemented transformation as {transformation_type}"
            )

        N = LL_.shape[0]
        # print("decode_side:",np.concatenate(
        #   [np.concatenate([LL_,HD_],axis=1),
        #   np.concatenate([VD_,DD_],axis=1),],axis=0
        # ))

        # -----
        # plaintext_chaotic_x=VDT_utils.calculate_y_from_x(LL_,x_i_arr[0])

        HD_VD_DD___ = np.concatenate(
            [
                HD_.reshape(N, N, -1),
                VD_.reshape(N, N, -1),
                DD_.reshape(N, N, -1),
            ],
            axis=2,
        )

        LL = cls.uncompute_plaintext_diffusion(
            LL_,
            chaotic_x=VDT_utils.calculate_y_from_x(HD_VD_DD___, x_i_arr[0]),
            chaotic_a=r_i_arr[0],
        )

        # print("decode_side_LL:",LL)
        # print("x_i_arr[0]=",x_i_arr[0],"chaotic_x=",plaintext_chaotic_x,"chaotic_a=",r_i_arr[0])

        HD_VD_DD = cls.unpermutate_cross_subband(
            HD_VD_DD___,
            r_i_arr[0],
            x_i_arr[0],
            r_i_arr[1:4],
            x_i_arr[1:4],
            existing_zigzag_path,
        )

        HD = HD_VD_DD[:, :, 0]
        VD = HD_VD_DD[:, :, 1]
        DD = HD_VD_DD[:, :, 2]

        if transformation_type == "vdt":
            image = VDT_utils.ivdt(LL, HD, VD, DD)
        elif transformation_type == "dwt":
            image = pywt.idwt2([LL, [HD, VD, DD]], "haar")
        elif transformation_type == "dct":
            image = VDT_utils.join_4_components_by_idct([LL, HD, VD, DD])
        elif transformation_type == "cumsum2d":
            image = VDT_utils.join_4_components_by_icumsum2d([LL, HD, VD, DD])
        elif transformation_type == "AES":
            image = VDT_utils.join_4_components_by_iAES(
                [LL, HD, VD, DD],
                extra_secret_key["password"],
                extra_secret_key["salt"],
                extra_secret_key["iv"],
            )
        elif transformation_type == "cumsum2d_AES":
            image = VDT_utils.inverse_cumsummation_2d(
                VDT_utils.join_4_components_by_iAES(
                    [LL, HD, VD, DD],
                    extra_secret_key["password"],
                    extra_secret_key["salt"],
                    extra_secret_key["iv"],
                )
            )
        elif transformation_type == "cumsum2d_VDT":
            image = VDT_utils.inverse_cumsummation_2d(VDT_utils.ivdt(LL, HD, VD, DD))
        else:
            raise ValueError(
                f"No such implemented transformation as {transformation_type}"
            )

        return image

    # @classmethod
    # def decode_layerwise_one_round(
    #     cls,
    #     cipher_image: np.ndarray,
    #     x_i_arr,
    #     r_i_arr,
    #     transformation_type: str,
    #     existing_zigzag_path: Optional[str],
    #     extra_secret_key: dict,
    # ):

    #     if cipher_image.ndim == 2:
    #         return cls.decode_one_round(
    #             cipher_image,
    #             x_i_arr,
    #             r_i_arr,
    #             transformation_type,
    #             existing_zigzag_path,
    #             extra_secret_key,
    #         )
    #     elif cipher_image.ndim == 3:
    #         channel_wise_images = [
    #             cls.decode_one_round(
    #                 cipher_image[:, :, i],
    #                 x_i_arr,
    #                 r_i_arr,
    #                 transformation_type,
    #                 existing_zigzag_path,
    #                 extra_secret_key,
    #             )
    #             for i in range(cipher_image.shape[2])
    #         ]

    #         return np.stack(channel_wise_images, axis=-1)
    #     else:
    #         raise NotImplementedError(f"{cipher_image.ndim}-dim not implemented.")

    @classmethod
    @timeit
    def decode(
        cls,
        cipher_image,
        secret_key,
        transformation_type="vdt",
        existing_zigzag_path: Optional[str] = None,
    ):
        (x1, r1), (x2, r2) = VDT_utils.key_schedule(secret_key[:-1])
        extra_key1 = {}
        extra_key2 = {}

        for extra_key_fragment in ["xor_of_chaos", "password", "salt", "iv"]:
            if extra_key_fragment not in secret_key[-1].keys():
                continue

            extra_key1[extra_key_fragment] = secret_key[-1][extra_key_fragment][0]
            extra_key2[extra_key_fragment] = secret_key[-1][extra_key_fragment][1]

        image_decoded_once = cls.decode_one_round(
            cipher_image, x2, r2, transformation_type, existing_zigzag_path, extra_key2
        )
        image_decoded_twice = cls.decode_one_round(
            image_decoded_once,
            x1,
            r1,
            transformation_type,
            existing_zigzag_path,
            extra_key1,
        )

        return image_decoded_twice


if __name__ == "__main__":

    cipher_image = np.load("data/outputs/cryptic_image.npy", allow_pickle=True)
    cipher_image = np.array(cipher_image, dtype="int32")

    transformation_type = "cumsum2d_VDT"

    with open("data/outputs/metadata.json", "r") as reader_f:
        json_data = json.load(reader_f)
        secret_key = json_data["secret_key"]

    deciphered_image = VdtMztDecoder.decode(
        cipher_image,
        secret_key,
        transformation_type=transformation_type,
        existing_zigzag_path="./data/zigzag",
    )

    plt.imshow(deciphered_image, cmap='gray')
    plt.title("Deciphered Image")
    plt.show()

    ## -----------------------------TESTS-------------------------------------- ##
    from VdtMztEncoder import VdtMztEncoder

    # # @TEST unapply_diffusion

    # orig_arr=np.random.randint(0,256,(100))
    # orig_arr=(np.random.random((100))*2-1)
    # index_arr=np.argsort(np.random.rand(*orig_arr.shape))

    # diffused_arr=VdtMztEncoder.apply_diffusion(orig_arr,index_arr,256)
    # dediffused_arr=VdtMztDecoder.unapply_diffusion(diffused_arr,index_arr,256)
    # assert np.all(dediffused_arr==orig_arr)
    # print("passed inverse diffusion!")

    # # @TEST uncompute_plaintext_diffusion

    # dummy_LL=np.random.randint(0,256,(256,256))
    # chaotic_x=np.random.random()
    # chaotic_a=np.random.random()
    # diffused_LL=VdtMztEncoder.compute_plaintext_diffusion(dummy_LL,chaotic_x,chaotic_a)
    # dediffused_LL=VdtMztDecoder.uncompute_plaintext_diffusion(diffused_LL,chaotic_x,chaotic_a)

    # assert(np.all(dediffused_LL==dummy_LL))
    # print("passed inverse plaintext!")

    # # @TEST unmodified_zigzag_permutation

    # orig_matrix=np.random.randint(0,256,(32,32))
    # chaotic_x=np.random.random()
    # chaotic_a=np.random.random()
    # permuted_matrix=VdtMztEncoder.modified_zigzag_permutation(orig_matrix,chaotic_x,chaotic_a,existing_zigzag_path=None)
    # unpermuted_matrix=VdtMztDecoder.unmodified_zigzag_permutation(permuted_matrix,chaotic_x,chaotic_a,existing_zigzag_path=None)

    # assert(np.all(unpermuted_matrix==orig_matrix))
    # print("passed inverse zigzag!")

    # # @TEST unpermute_cross_subband

    # orig_matrix=np.random.randint(0,256,(32,32,3))
    # r0=np.random.random()
    # x0=np.random.random()
    # r_array=np.random.random(3)
    # x_array=np.random.random(3)

    # permuted_matrix=VdtMztEncoder.permutate_cross_subband(orig_matrix,r0,x0,r_array,x_array,existing_zigzag_path=None)
    # unpermuted_matrix=VdtMztDecoder.unpermutate_cross_subband(permuted_matrix,r0,x0,r_array,x_array,existing_zigzag_path=None)

    # assert(np.all(unpermuted_matrix==orig_matrix))
    # print("passed inverse cross subband!")

    # # @TEST decode_one_round
    # image=np.random.randint(0,256,(32,32))
    # r_array=np.random.random(5)
    # x_array=np.random.random(5)
    # cipher_image=VdtMztEncoder.encode_one_round(image, x_array,r_array,transformation_type='vdt',existing_zigzag_path=None)
    # deciphered_image=VdtMztDecoder.decode_one_round(cipher_image, x_array,r_array,transformation_type='vdt',existing_zigzag_path=None)

    # assert(np.all(deciphered_image==image))
    # print("passed decode_one_round!")

    # @TEST decode

    # secret_key=(
    #   np.random.random(), # x0_1
    #   np.random.random(), # x0_2
    #   np.random.random()*128, # r0_1
    #   np.random.random()*128, # r0_2
    #   np.random.randint(0,128,(4,)), # a1
    #   np.random.randint(0,128,(4,)), # a2
    #   np.random.random()*128, # t0_1
    #   np.random.random()*128, # t0_2
    #   {} # extra_secret_key
    # )
    # image=np.random.randint(0,256,(32,32))
    # cipher_image=VdtMztEncoder.encode(image,
    #                                   secret_key,
    #                                   transformation_type='vdt',
    #                                   existing_zigzag_path='./data/zigzag'
    #                                   )
    # deciphered_image=VdtMztDecoder.decode(cipher_image,
    #                                       secret_key,
    #                                       transformation_type='vdt',
    #                                       existing_zigzag_path='./data/zigzag'
    #                                       )

    # assert(np.all(deciphered_image==image))
    # print("passed decode!")

    # image_3d=np.random.randint(0,256,(32,32,3))
    # cipher_image_3d=VdtMztEncoder.encode(image_3d,
    #                                   secret_key,
    #                                   transformation_type='vdt',
    #                                   existing_zigzag_path='./data/zigzag'
    #                                   )
    # deciphered_image_3d=VdtMztDecoder.decode(cipher_image_3d,
    #                                       secret_key,
    #                                       transformation_type='vdt',
    #                                       existing_zigzag_path='./data/zigzag'
    #                                       )

    # assert(np.all(deciphered_image_3d==image_3d))
    # print("passed decode 3d!")
