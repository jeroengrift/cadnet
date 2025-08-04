def mean_std_pixel_per_channel(loader):
    # https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/#:~:text=Image%20normalization%20is%20the%20process,dividing%20by%20the%20standard%20deviation.
    # Compute the mean and standard deviation of all pixels in the dataset, batch_size must be 1 (for correct calculation)
    num_images = len(loader)
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for images, _, __ in loader:
        batch_size, num_channels, height, width = images.shape
        for c in range(num_channels):
            mean[c] += images.squeeze()[c,:,:].cpu().detach().numpy().mean(axis=(0, 1))
            std[c] += images.squeeze()[c,:,:].cpu().detach().numpy().std(axis=(0, 1))

    print(mean)
    print(std)
    print(num_images)

    mean = [val / num_images for val in mean]
    std = [val / num_images for val in std]

    return mean, std
