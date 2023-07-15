import numpy as np
from PIL import Image

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image


def test(batch_size):
    data = load_images('/kaggle/working/test', batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights(
        '/kaggle/working/kaggle/working/training/715/generator_3_2811.h5')
    generator_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generator_images])

    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generator_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))


def test_command(batch_size):
    return test(batch_size)


if __name__ == '__main__':
    test_command(4)
