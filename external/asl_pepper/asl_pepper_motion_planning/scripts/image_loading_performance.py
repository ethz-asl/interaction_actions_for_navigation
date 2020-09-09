import os
import skimage
from timeit import default_timer as timer

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

from pympler import asizeof

MAX_BATCH_SIZE = 1000
IMAGE_DIR = os.path.expanduser('/media/pithos/nyc_frames')
MASK_DIR = os.path.expanduser('/media/pithos/nyc_masks')
filenames = sorted(os.listdir(IMAGE_DIR))

def get_memusage():
    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0]
        memsize = int(memusage.split(' ')[-2])
        memunit = memusage.split(' ')[-1]
        if not memunit == "kB":
            raise ValueError(memunit + " unknown")
    return memsize

start_memusage = get_memusage()
memunit = "kB"
print("Initial memory: {} {}".format(start_memusage, memunit))
printed_lines = 1

for batch_size in range(1, MAX_BATCH_SIZE):
    if batch_size > 100 and not batch_size % 10 == 0:
        continue
    if batch_size > 1000 and not batch_size % 100 == 0:
        continue
    try:
        filename_batch = filenames[:batch_size]
        tic = timer()
        image_batch = [skimage.io.imread(os.path.join(IMAGE_DIR, filename)) for filename in filename_batch]
        toc = timer()
        load_time = toc-tic
        size = get_memusage()
        if printed_lines == 1 or printed_lines % 32 == 0:
            print("# images loaded | loading time [s] | per image [s] | size [{}] | diff [{}] | per image [{}]".format(memunit, memunit, memunit))
            print("-------------------------------------------------------------------------------------------")
        print(" {:>14} | {:>16.2f} | {:>13.3f} | {:>9} | {:>9} | {:>10} ".format(
            len(image_batch), load_time, load_time/len(image_batch), size, size-start_memusage, (size-start_memusage)//len(image_batch)))
        printed_lines += 1
        del image_batch
    except KeyboardInterrupt:
        break
