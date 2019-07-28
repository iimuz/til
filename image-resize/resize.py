from PIL import Image


def resize():
    img = Image.open('newblocks.png')
    img = img.resize((80, 80), Image.LANCZOS)
    img.save('newblocks-small.png')


if __name__ == '__main__':
    resize()
