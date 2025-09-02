import matplotlib.pyplot as plt

def load_image(path):
    """Load an image as numpy array
    parameters:
    -----------
    path: to the image
    """
    return plt.imread(path)