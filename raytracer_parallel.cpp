//Original code was found here https://github.com/MarcusMathiassen/BasicRaytracer30min
//We added all comments explaining the code, as well as making the scene bigger and more complex.

/**
 * @file raytracer.cpp
 * @author Adam Kenney, Jaden Stith... Credits to original code above
 * @brief This program renders a simple scene using a basic raytracing algorithm on an 800 by 800 pixel map. 
 * @version 0.1
 * @date 2021-12-01
 * 
 * 
 * 
 */
#include <fstream>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <omp.h>
typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

/*
 * This struct is used to create a vector with three dimensions, define basic operations one can perform on vectors
 */
struct Vec3 {
  double x,y,z;
  //Constructor
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

  //operations
  Vec3 operator + (const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
  Vec3 operator - (const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
  Vec3 operator * (double d) const { return Vec3(x*d, y*d, z*d); }
  Vec3 operator / (double d) const { return Vec3(x/d, y/d, z/d); }

  //This function returns the normalized vector, or in other terms the vector divided by it's magnitude
  Vec3 normalize() const {
    double mg = sqrt(x*x + y*y + z*z);
    return Vec3(x/mg,y/mg,z/mg);
  }
};

//This function computes the dot product of two vectors a and b
inline double dot(const Vec3& a, const Vec3& b) {
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

/**
 * The Ray struct creates a ray extendending from origin o in direction d.
 * d is a normalized vector that points in the direction relative to the point. o + d gives you a ray that starts at the point of origin and 
 * extends infinitely in the desired direction.
 *  
 */
struct Ray {
  Vec3 o,d;
  Ray(const Vec3& o, const Vec3& d) : o(o), d(d) {}
};

/**
 * The sphere struct is used to construct a sphere in the scene. It consists of a vector pointing to the center, and a radius.
 * 
 */
struct Sphere {
  Vec3 c;
  double r;
  Vec3 col;
  //Constructor
  Sphere(const Vec3& c, double r, const Vec3& col) : c(c), r(r), col(col) {}

  //This function returns the normal vector to the point of intersect on the sphere.
  Vec3 getNormal(const Vec3& pi) const { return (pi - c) / r; }

  //This function calculates if a ray intersects the sphere. It returns true and false, and it leaves the distance from the point of origin o
  //the closest point of intersect in the variable t.
  bool intersect(const Ray& ray, double &t) const {
    const Vec3 o = ray.o;
    const Vec3 d = ray.d;
    const Vec3 oc = o - c;
    const double b = 2 * dot(oc, d);
    const double c = dot(oc, oc) - r*r;
    double disc = b*b - 4 * c;    //Calculate the discriminate. Keep in mind a is dot(d, d) which is 1
    if (disc < 1e-4) return false; //No intersect
    if (disc == 0){                //Only 1 intersect
      t = -b;
      return true;
    }else{                         //2 intersects        
      disc = sqrt(disc);
      const double t0 = -b - disc;
      const double t1 = -b + disc;
      t = (t0 < t1) ? t0 : t1;    //t must be the intersect that is closest to the point of origin.
      return true;
    }
  }
};

//This method caps the rgb values of the color at 255.
void clamp255(Vec3& col) {
  col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
  col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
  col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main() {
  omp_set_num_threads(2);
  const int H = 800;    //Height of the image in pixels
  const int W = 800;    //Width of the image in pixels

  const Vec3 white(255, 255, 255);    //RGB value for white
  const Vec3 black(0, 0, 0);          //RGB value for black
  const Vec3 blue(50, 160, 240);      //RGB value for lightish blue
  const Vec3 purple(210, 50, 235);    //RGB value for purple
  const Vec3 green(0, 255, 0);        //RGB vlaue for green

  const Sphere sphere1(Vec3(W*0.5, H*0.5, 50), 50, blue);  //Initializes a sphere of radius 50 in the middle of the screen
  const Sphere sphere2(Vec3(W*.25, H*.25, 30), 60, purple);  //Initializing other spheres in the scene
  const Sphere sphere3(Vec3(W*.75, H*.75, 20), 20, green);
  const Sphere light(Vec3(W*0.5, 0, 50), 1, white);             //Initializes a sphere that represents the light source
  Sphere objects[] = {sphere1, sphere2, sphere3};             //Array to hold all the objects in our scene

  //establishes a connection to our out.ppm file which is a pixel map that will be used to draw our image
  std::ofstream out("out.ppm");
  out << "P3\n" << W << ' ' << H << ' ' << "255\n";

  //t is essentially the distance from the point of origin for a ray to the point of intersect. It is valued by the function sphere.intersect
  double t;
  Vec3 pix_col(black);

  timestamp_t t0 = get_timestamp();
  //Loop through all H*W pixels

  for (int y = 0; y < H; ++y) {
    #pragma omp parallel for shared(objects)
    for (int x = 0; x < W; ++x) {
      //Default color is black, unless there is an intersect
      pix_col = black;

      //Cast a ray from each pixel straight out in the z direction.
      const Ray ray(Vec3(x,y,0),Vec3(0,0,1));

      //If there is an intersect, then we will change the color of pixel to reflect its distance and angle from the lightsource.
      for(Sphere s : objects){
        if (s.intersect(ray, t)) {
          const Vec3 pi = ray.o + ray.d*t;        //point of intersect (origin + (direction * t))
          const Vec3 L = light.c - pi;            //Luminence vector which goes from the point of intersection in the direction of the light source
          const Vec3 N = s.getNormal(pi);         //the sphere's normal vector from the point of intersect
          const double dt = dot(L.normalize(), N.normalize());

          pix_col = (s.col + white*dt) * 0.5; 
          clamp255(pix_col);
        }
      }
      out << (int)pix_col.x << ' '
          << (int)pix_col.y << ' '
          << (int)pix_col.z << '\n';
    }
  }
  timestamp_t t1 = get_timestamp();
  std::cout << (t1-t0) / 1000000.0L << std::endl;
}