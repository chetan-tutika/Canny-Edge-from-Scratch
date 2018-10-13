# Canny-Edge-fro-Scratch
To implement canny edge detection without using opencv functions

### Magnitude
Gives the combination of x and y gradients of the image
### Non Maximal Suppression
Suppresses the thick edges to thin edges using non-maximal-suppression
### Edge Linking
Eliminate the unwanted edges and to retain the weak edges connected to the strong edges
Strong edges are edges with intensity greater than median + 2*(standard deviation)
Weak edges correspond to edges with intensity inbetween  (median + 2*(standard deviation) and median/(10^3))
median and standard deviation are calculated on the nms output

## Results
### Input
![135069](https://user-images.githubusercontent.com/41950483/46907816-68273500-cee6-11e8-91b0-3a6c50133e4b.jpg)
### Magnitude
![mag](https://user-images.githubusercontent.com/41950483/46907822-7bd29b80-cee6-11e8-8465-bd71c605744c.png)
### Final Result
![final](https://user-images.githubusercontent.com/41950483/46907826-855c0380-cee6-11e8-992f-645dd99dba6d.png)

Run the edgeLink.py for the final solution

