B = []
N = ((32, 64, 128), (16, 0, 1), (8, 4, 2))
A0 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124,
      126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
      225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254]
A1 = [7, 14, 28, 56, 112, 131, 193, 224]
A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240]
A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193, 195, 199, 224, 225, 227, 240, 241, 248]
A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 193, 195, 199, 207, 224, 225, 227,
      231, 240, 241, 243, 248, 249, 252]
A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 191,
      193, 195, 199, 207, 224, 225, 227, 231, 239, 240, 241, 243, 248, 249, 251, 252, 254]
A1pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124, 126,
         127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
         231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254]
pix = lambda x: 0 if x == 255 else 1


def thinner(im, W):
    for x in range(1, im.size[0] - 1):
        for y in range(1, im.size[1] - 1):
            weight = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    weight += N[i + 1][j + 1] * pix(im.getpixel((x + i, y + j)))
            if weight in W:
                im.putpixel((x, y), 255)
    return im


def phase(im, B, W):
    for b in B:
        weight = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                weight += N[i + 1][j + 1] * pix(im.getpixel((b[0] + i, b[1] + j)))
        if weight in W:
            im.putpixel(b, 255)
            B.remove(b)
    return B


def border(im, A0):
    B = {}
    for x in range(1, im.size[0] - 1):
        for y in range(1, im.size[1] - 1):
            bit = pix(im.getpixel((x, y)))
            if bit == 0: continue
            # Weight
            weight = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    weight += N[i + 1][j + 1] * pix(im.getpixel((x + i, y + j)))
            if weight in A0:
                B[(x, y)] = 1
    return B


def skeletize(im):
    flag = True
    while flag:
        B = border(im, A0)
        Bp1 = phase(im, list(B), A1)
        Bp2 = phase(im, Bp1, A2)
        Bp3 = phase(im, Bp2, A3)
        Bp4 = phase(im, Bp3, A4)
        Bp5 = phase(im, Bp4, A5)
        plist = Bp5
        if len(B) == len(plist): flag = False
    return thinner(im, A1pix)
