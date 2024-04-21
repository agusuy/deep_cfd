import matplotlib.pyplot as plt
import numpy as np

def get_color_map():
    color_map = plt.cm.jet
    color_map.set_bad(color="black")
    return color_map

def preprocess_frame(frame, min_val=0.0, max_max=1.0):
    frame[frame<min_val] = np.nan
    frame[frame>max_max] = np.nan
    
def plot_frame(frame, title, color_map=None, show_color_var=True):
    if not color_map:
        color_map = get_color_map()
    
    plt.clf()
    plt.imshow(np.squeeze(frame).transpose(), cmap=color_map)
    plt.title(title)
    plt.axis("off")
    if show_color_var:
        plt.colorbar(label="Velocity", orientation="horizontal")
    plt.show()

def plot_compare_frames(frame1, frame2, title1, title2, color_map=None, show_color_var=True, orientation="v"):
    if not color_map:
        color_map = get_color_map()

    if orientation == "v":
        _, axes = plt.subplots(1, 2, figsize=(10, 20))
    elif orientation == "h":
        _, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].imshow(np.squeeze(frame1).transpose(), cmap=color_map)
    axes[0].title.set_text(title1)
    axes[0].axis("off")

    axes[1].imshow(np.squeeze(frame2).transpose(), cmap=color_map)
    axes[1].title.set_text(title2)
    axes[1].axis("off")

def plot_frame_histogram(frame, title):
    plt.clf()
    plt.hist(frame.flatten(), bins='auto')
    plt.title(title)
    plt.xlabel('Velocity')
    plt.ylabel('Frequency')
    plt.show()
