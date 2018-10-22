import numpy as np
from skimage import io, color
from matplotlib import pyplot as plt

# Configure plot to show imaging results.
plt.figure(figsize=(9,6))
plt.title("Main Image")
im = io.imread('Dog.jpg')
#im = color.rgb2lab(rgb)
print(type(im))
plt.imshow(im)
plt.show()

# Split main image into three separate pieces. 
r,g,b = im[:,:,0], im[:,:,1], im[:,:,2]

# Perform SVD on the three separate components.
Ur, Sr, VTr = np.linalg.svd(r)
Ug, Sg, VTg = np.linalg.svd(g)
Ub, Sb, VTb = np.linalg.svd(b)

for i in range(0, 400, 50):

	# Apply rank selection on the individual components.
	ranked_r = np.array(np.matrix(Ur[:, :i]) * np.diag(Sr[:i]) * np.matrix(VTr[:i, :]))
	ranked_g = np.array(np.matrix(Ug[:, :i]) * np.diag(Sg[:i]) * np.matrix(VTg[:i, :]))
	ranked_b = np.array(np.matrix(Ub[:, :i]) * np.diag(Sb[:i]) * np.matrix(VTb[:i, :]))

	# Stack r,g,b ranks into single matrix of mxnx3 dimensions
	rgb = np.dstack((ranked_r,ranked_g,ranked_b)).astype('uint8')

	plt.imshow(rgb)
	plt.title("Ranked {} - Dog".format(i))
	plt.show()
