
fig, ax = plt.subplots(1,2)

ax[0].imshow(val_inputs_image[0,1,:,:])
ax[1].imshow(val_inputs_image2[0,1,:,:]*adj.transpose())
    