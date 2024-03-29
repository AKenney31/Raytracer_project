# Raytracer_project
Authors: Adam Kenney, Jaden Stith, Trevor Koenig

## File Description/How To Compile
- raytracer.cpp: This is the original serial code we made. Compile with "g++ raytracer.cpp"
- raytracer_refactored.cpp: This is the serial that we refactored to be able to work with parallelization. The other version writes directly into the out.ppm file from the for loop and there is no way known by us to make the threads put the pixels into out.ppm in the correct order. So this file contains a massive 2D array of pixels that stores each pixel in the correct position. Compile with "g++ raytracer_refactored.cpp"
- raytracer_parallel.cpp: This is the parallelized version of our raytracer. Compile with "g++ raytracer_parallel.cpp -fopenmp"
- raytracer.cpp will run on any machine with any size window width and height, the other two will work best on a super computer unless you decrease the width and height of the output window because the array is too large to handle. 

## Project Description
This is our parallel programming project in which we have to use real world serial code and attempt to parallelize it. The project we chose to try is a raytracer. Raytracing is a basic method to create 3 dimensional graphics. It works by shooting rays out into a scene and if it makes contact with an object in the scene, it calculates what shade the pixel should be based on the position of the lightsource. We made an orthographic raytracer, meaning that it shoots a ray straight out in the z direction from each pixel in the window. This is less efficient than using the camera approach to this problem when you shoot rays out from a camera's point of view. The orthographic approach is better suited for our project because it is effected by the size of the window. So we start with 800 X 800 which performs pretty well, and then we bump the window size to 10000 X 10000 to see how the serial version will perform compared to the parallel version. Keep in mind that on the 10000 X 10000 version, you will not be able to view the ppm file, but it will still run all the way and produce an output so we can still measure it's performance. The parallel version was made using openmp to parallelize over CPUs. We ran the code on the supercomputer Bridges 2 located in Pennsylvania.

## Experimental Steps
- Build serial code that renders a couple of spheres on a 800x800 pixel window. Then attempt to parallelize the code to the best of our ability.
- We parallelize using openmp
- In the code we time the code using time.h, and output this in the console
- Once we have recorded the time for the serial and parallel code, and prove that the rendered image is the same, we will bump the window size to 10000x10000 pixels and repeat We will record the times for the parallel version using 2, 4, 8, 16 cores.

## References
Original serial code was found here https://github.com/MarcusMathiassen/BasicRaytracer30min
We added all comments explaining the code, as well as making the scene bigger and more complex, change some of the existing code to better fit our needs, and we added all the code to get and print the execution time.

