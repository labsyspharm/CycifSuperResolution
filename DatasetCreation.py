import sklearn
import tifffile
import argparse
import sys
import os
import pickle
import numpy
import tqdm

def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("input", type=str, help="Path to directories with image")
    output.add_argument("output", type=str, help="Path to store results")
    output.add_argument("--recursive", action="store_true", help="Iterate through directories.")
    output.add_argument("--file-suffix", type=str, help="Only select files with this sufix. For testing is \"stack.tif\"", default=None)
    output.add_argument("--norm-type", choices=["std", "gauss", "no"], help="How to normalize images.\nStd is bit normalization.\nGauss scales the image so the mean is `80% of bit range and then bit normalizes.", default="std")
    output.add_argument("--prefix", type=str, help="Prefix for generated files", default="")
    output.add_argument("--transformations", action="store_true", help="Do transformations to images")

    output = output.parse_args(sys.argv[1:])

    if not os.path.isdir(output.input):
        print("Path {} is not directory.".format(output.input))

    if not os.path.exists(output.output):
        os.makedirs(output.output, exist_ok=True)
    if not os.path.exists(os.path.join(output.output, "input")):
        os.makedirs(os.path.join(output.output, "input"), exist_ok=True)
    if not os.path.exists(os.path.join(output.output, "output")):
        os.makedirs(os.path.join(output.output, "output"), exist_ok=True)

    return output


def save_combination(args, input_image, output_image, name):
    with open(os.path.join(args.output, "input", name), "wb") as f:
        pickle.dump(input_image, f)

    with open(os.path.join(args.output, "output", name), "wb") as f:
        pickle.dump(output_image, f)


def normalize_8_bit(args, image):
    if image.dtype == numpy.int8 or image.dtype == numpy.uint8:
        return image / (2**8)  # tecnically not necessary but for completion-wise
    elif image.dtype == numpy.float16 or image.dtype == numpy.uint16:
        return image / (2**16)
    elif image.dtype == numpy.float32 or image.dtype == numpy.uint32:
        return image / (2**32)
    elif image.dtype == numpy.float64 or image.dtype == numpy.uint64:
        return image / (2**64)
    else:
        raise Exception("Invalid dtype {}".format(image.dtype))


def normalize(args, image):
    if args.norm_type == "std":
        return normalize_8_bit(args, image) * 255.0
    elif args.norm_type == "gauss":
        """
        3 component normally separates background noise from nuclei so normalize the top mean to something high
        and then clip over values
        """
        gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
        gmm.fit(numpy.log10(image.reshape((-1, 1))))
        mean = max(10 ** gmm.means_[:, 0])
        bits = 8 * image.itemsize
        dtype = image.dtype  # it is important to mantain dtype for normalization later

        tif = (0.8 * 2 ** bits) * image / mean  # 80% of 16 bits is ~52000
        tif[tif > 2 ** bits] = 2 ** bits
        return normalize_8_bit(args, tif.astype(dtype)) * 255.0
    elif args.norm_type == "no":
        return image


def handle_file(args, file_path):
    tiff = tifffile.imread(file_path)
    if tiff.shape[-1] != 512:
        print(file_path, "Not 512 actually", tiff.shape[-1])
        return

    tiff = normalize(args, tiff)

    input_image = tiff[0]

    for i in range(1, len(tiff)):
        save_combination(args, input_image, tiff[i], args.prefix + "_" + str(i) + "_" + file_path.split("/")[-1])

    if args.transformations:
        # rotations
        for rot in range(3):
            tiff = numpy.rot90(tiff, 1, axes=(1, 2))
            input_image = tiff[0]

            for i in range(1, len(tiff)):
                save_combination(args, input_image, tiff[i],
                                 args.prefix + "_rotated_" + str(rot) + "_" + str(i) + "_" + file_path.split("/")[-1])
                save_combination(args, numpy.flip(input_image, 0), numpy.flip(tiff[i], 0),
                                 args.prefix + "_rotated_hflip_" + str(rot) + "_" + str(i) + "_" +  file_path.split("/")[-1])
                save_combination(args, numpy.flip(input_image, 1), numpy.flip(tiff[i], 1),
                                 args.prefix + "_rotated_vflip_" + str(rot) + "_" + str(i) + "_" +  file_path.split("/")[-1])


def pipeline(args):
    if args.recursive:
        for dir in tqdm.tqdm(os.listdir(args.input), desc="Directory: "):
            for f in tqdm.tqdm(os.listdir(os.path.join(args.input, dir)), leave=True, desc="Files: "):
                if os.path.isfile(os.path.join(args.input, dir, f)) and \
                        args.file_suffix is not None and \
                        f.endswith(args.file_suffix):
                    handle_file(args, os.path.join(args.input, dir, f))
    else:
        for f in tqdm.tqdm(os.listdir(args.input), desc="Files: "):
            if os.path.isfile(os.path.join(args.input, f)) and \
                    args.file_suffix is not None and \
                    f.endswith(args.file_suffix):
                handle_file(args, os.path.join(args.input, f))


if __name__ == "__main__":
    args = parse_args()
    pipeline(args)
