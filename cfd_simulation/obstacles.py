from math import cos, sin, pi

def circumference(center_x, center_y, radius):
    return lambda x, y: (x-center_x)**2+(y-center_y)**2 < radius**2

def ellipse(center_x, center_y, semi_major_axis, semi_minor_axis_proportion, degrees=0):
    
    # convert degrees to radians
    alpha = degrees * (pi/180)
    
    semi_minor_axis = semi_major_axis * semi_minor_axis_proportion

    return lambda x, y: (
        (((cos(alpha)*(x-center_x)+sin(alpha)*(y-center_y))**2) 
         / (semi_major_axis**2))
        + (((sin(alpha)*(x-center_x)-cos(alpha)*(y-center_y))**2) 
           / (semi_minor_axis**2))
    ) < 1
