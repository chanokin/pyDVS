import cv2
import numpy



circle_filename = "white-circle-filled.png"
ring_filename = "white-ring.png"
circle = cv2.cvtColor(cv2.imread(circle_filename), cv2.COLOR_RGB2GRAY)
ring = cv2.cvtColor(cv2.imread(ring_filename), cv2.COLOR_RGB2GRAY)

start_scale = 1.0
end_scale = 0.05
scale_step = 0.005

num_steps = int( (start_scale - end_scale)/scale_step )
print(num_steps)

step_count = 0
out = numpy.zeros_like(circle, dtype=numpy.uint8)
fr_r = 0
to_r = 0
orig_w = circle.shape[0]
scl_w = 0
s = start_scale
for step_count in range(num_steps):
#for s in range(start_scale, end_scale - scale_step, -scale_step):
    scl_w = int(orig_w*s)
    fr_r = int( (orig_w - scl_w)/2. )
    to_r = fr_r + scl_w
    scaled = cv2.resize(circle, (scl_w, scl_w))
    out[:] = 0
    out[fr_r:to_r, fr_r:to_r] = scaled
    cv2.imwrite("circle_images/scaled-%010d-%s"%\
                (num_steps - step_count, circle_filename),
                out)
    scaled = cv2.resize(ring, (scl_w, scl_w))
    out[:] = 0
    out[fr_r:to_r, fr_r:to_r] = scaled
    cv2.imwrite("ring_images/scaled-%010d-%s"%\
                (num_steps - step_count, ring_filename),
                out)
    s -= scale_step
    