import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as Patches

# set color maps for visulization
# colormap = mpl.cm.Paired.colors
# num_color = len(colormap)

colormap = mpl.cm.Paired.colors
colormap = (
    (0.6509803921568628, 0.807843137254902, 0.8901960784313725), 
    (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
    (0.984313725490196, 0.6039215686274509, 0.6), 
    (0.8901960784313725, 0.10196078431372549, 0.10980392156862745), 
    (0.9921568627450981, 0.7490196078431373, 0.43529411764705883), 
    (1.0, 0.4980392156862745, 0.0), 
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098), 
    (0.41568627450980394, 0.23921568627450981, 0.6039215686274509), 
    (1.0, 1.0, 0.6), 
    (0.6941176470588235, 0.34901960784313724, 0.1568627450980392))

num_color = len(colormap)


def show_polygons(image, polys):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.fill(polygon[:,0], polygon[:, 1], color=color, alpha=0.3)
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    plt.show()

def save_viz(image, polys, save_path, filename):
    plt.axis('off')
    # plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    impath = osp.join(save_path, filename)
    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0)
    plt.clf()
    
def plot_features(fm, out_file):
    # https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
    feature_maps = fm.squeeze(0).data.cpu().numpy()
    fig = plt.figure(figsize=(30, 50))
    num_maps = len(feature_maps)
    for i in range(num_maps):
        a = fig.add_subplot(int(num_maps / 4) if num_maps > 4 else 1, 4, i+1)
        imgplot = plt.imshow(feature_maps[i])
        a.axis("off")
        a.set_title(str(i), fontsize=30)
        plt.colorbar(label="Score", orientation="vertical", fraction=0.046, pad=0.04) 
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



