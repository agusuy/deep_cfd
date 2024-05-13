import matplotlib.pyplot as plt

def plot_frame_histogram(frame, title):
    plt.clf()
    plt.hist(frame.flatten(), bins='auto')
    plt.title(title)
    plt.xlabel('Velocity')
    plt.ylabel('Frequency')
    plt.show()

