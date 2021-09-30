import imageio
import glob
anim_file = 'style_transfer.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('training_images/*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    [writer.append_data(image) for _ in range(8)]
  image = imageio.imread(filename)
  [writer.append_data(image) for _ in range(8)]
