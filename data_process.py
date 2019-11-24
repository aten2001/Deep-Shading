import OpenEXR
import Imath
import numpy as np

def exr_loader(path, ndim=3):
    """
    loads an .exr file as a numpy array
    :param path: path to the file
    :param ndim: number of channels that the image has,
                    if 1 the 'R' channel is taken
                    if 3 the 'R', 'G' and 'B' channels are taken
    :return: np.array containing the .exr image
    """

    # read image and its dataWindow to obtain its size
    pic = OpenEXR.InputFile(path)
    dw = pic.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 1:
        # transform data to numpy
        channel = np.fromstring(channel=pic.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        return np.array(channel)
    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.fromstring(pic.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match numpy style
        return np.array(allchannels).transpose((1, 2, 0))

# load all images listed in one txt file in "dataset" folder
def load_one_file(filename, dim):
    num_success = 0
    with open(filename) as f:
        images = []
        for line in f.readlines():
            path = line.split()[0]
            try:
                images.append(exr_loader(path, ndim=dim))
                num_success += 1
            except:
                print("Fail loading " + path)
    print ("Successfully loaded " + str(num_success) + " images from " + filename)
    return np.array(images)

# task = "AO", "GI",...
# buffers = a list of "position", "normal", "groundtruth",...
# return a dict of {"position" : np.array, ...}
def load_train(task, buffers=None):
    result = {}
    dims_for_channel = {
        "glossiness": 1
    } # default is 3

    if task == "AO":
        if buffers is None:
            buffers = ["position", "normal", "groundtruth"]
        for buffer_type in buffers:
            path = "./dataset/training_" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))

    elif task == "GI":
        if buffers is None:
            buffers = ["position", "normal", "light", "ground_truth"]
        for buffer_type in buffers:
            path = "./dataset/training_" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))

    elif task == "IBL":
        if buffers is None:
            buffers = ["camera", "normal", "diffuse", "ground_truth", "glossiness", "specular"]
        for buffer_type in buffers:
            path = "./dataset/" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))
    return result

def load_test(task, buffers=None):
    result = {}
    dims_for_channel = {
        "glossiness": 1
    } # default is 3

    if task == "AO":
        if buffers is None:
            buffers = ["position", "normal", "groundtruth"]
        for buffer_type in buffers:
            path = "./dataset/test_" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))

    elif task == "GI":
        if buffers is None:
            buffers = ["position", "normal", "light", "groundtruth"]
        for buffer_type in buffers:
            path = "./dataset/test_" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))

    elif task == "IBL":
        if buffers is None:
            buffers = ["camera", "normal", "diffuse", "groundtruth", "glossiness", "specular"]
        for buffer_type in buffers:
            path = "./dataset/test_" + buffer_type + ".txt"
            result[buffer_type] = load_one_file(path, dims_for_channel.get(buffer_type, 3))
    return result