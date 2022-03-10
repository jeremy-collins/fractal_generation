# fractal_generation
Two varieties of fractals are generated in this repository.

First, we have the Dragon Fractal. This is the shape generated by a thin strip of paper if you were to fold it in half repeatedly, and then unfold all of your folds to 90 degree angles. My inspiration to pursue this project came when I was playing with a straw wrapper at a restaurant.

![Alt text](backgrounds/dragon_fractal_3_cropped.png?raw=true "Title")

This fractal was generated using Python Turtle graphics as the framework. Essentially, I recursively generate a string of left and right turns which commands a line-drawing agent to trace out the entire fractal.

Next, we have the classic Mandelbrot fractal. This fractal was generated by starting with a complex number $c$ between -$2 - 2i$ and $2 + 2i$. This complex number is then plugged into the recurrence relation: $z_{i+1} = z_i^2 + c$, where z is the previous output (initialized to 0), and c is the point we are evaluating. A point is \textit{in} the Mandelbrot set if z does not explode to infinity within a given depth.

![Alt text](backgrounds/mandelbrot_7_cropped.png?raw=true "Title")

Simply drag your mouse across the screen to make a zoom window and generate a new plot! The depth and resolution are completely configurable.
