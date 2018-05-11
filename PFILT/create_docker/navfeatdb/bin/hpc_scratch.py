
def get_img(img_num):
    import pytables as tb

    flight = tb.open_file('/home/venabled/data/uvan/fc2_f5.hdf', 'r')
    images = flight.root.camera.image_raw.compressed.images
    flight.close()
    return images[img_num]
